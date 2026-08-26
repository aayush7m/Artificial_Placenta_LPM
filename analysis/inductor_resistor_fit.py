"""
Fit a series resistor + inductor (R, L) pair for each passive AP component.

Each AP component (ECMO, restrictor, tubing) is currently a pure resistor.
Here it is modelled as an R-L series element:

        dP(t) = R * Q(t) + L * dQ/dt

Given the measured pressure drop dP(t) across a component, the flow it
predicts is the solution of the first-order ODE

        L * dQ/dt + R * Q = dP(t)          <=>   Q(s)/dP(s) = 1/(L*s + R)

(R, L) are optimised so that the predicted flow waveform matches the mean
value AND the amplitude of the clinical flow waveform:

        R  <-> sets the mean (DC gain 1/R)
        L  <-> sets the amplitude (attenuates the pulsatile component)

Pressure taps bounding each component (per the AP circuit
UA -> Pump -> ECMO -> Restrictor -> UV):

        pump        : Pre-Pump          -> Pre-ECMO          (not fitted here)
        ECMO        : Pre-ECMO          -> Post-ECMO
        restrictor  : Post-ECMO         -> Post-Restrictor
        tubing      : Post-Restrictor   -> UV

UV is NOT instrumented in the clinical trial (the 'Jugular Vein Pressure #2'
channel is flat/dead over this window), so the UV pressure is taken from a
30 s run of the AP_model_v12_6 LPM.  See UV_SOURCE below.

Units: pressure [mmHg], flow [mL/s]  =>  R [mmHg*s/mL], L [mmHg*s^2/mL].
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import lsim

# Bootstrap: put the project root on sys.path so this file runs directly
# (python3 analysis/inductor_resistor_fit.py) from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This fit was built against v12_6 (see README lineage) and is kept pinned to
# it so the published R-L values stay reproducible.
from superseded.AP_model_v12_6 import (
    START_TIME,
    END_TIME,
    ClinicalData,
    ArtificialPlacentaModel,
)
from project_paths import RESULTS_AP_RL_FIT, clinical_csv, ensure

# ============================================================
#  SETTINGS
# ============================================================
SIM_DURATION_S = 30.0      # model run length used for the UV reference pressure

# How the (unmeasured) UV pressure bounding the tubing is obtained:
#   'model_mean'     -> constant, = time-average of P_UV from the v12_6 run
#   'model_waveform' -> full P_UV(t) from the v12_6 run, resampled onto the
#                       clinical grid.  NOTE: the model heart rate is not
#                       phase-locked to the clinical trace, so this introduces
#                       a spurious beat into dP_tubing.  Use with care.
#   <float>          -> user-supplied constant UV pressure in mmHg
UV_SOURCE = 'model_mean'

SETTLE_S = 3.0             # discard this much of the predicted flow (ODE start-up)
AMP_LO_PCT, AMP_HI_PCT = 5.0, 95.0   # robust peak-to-peak definition

RESULTS_JSON = os.path.join(ensure(RESULTS_AP_RL_FIT), 'ap_rl_fit_results.json')
# ============================================================


def amplitude(x):
    """Robust peak-to-peak amplitude (outlier-insensitive)."""
    return np.percentile(x, AMP_HI_PCT) - np.percentile(x, AMP_LO_PCT)


def predict_flow(R, L, dP, t):
    """Flow predicted by an R-L series element driven by dP(t).

    Solves  L*dQ/dt + R*Q = dP(t)  with Q(0) at the resistive steady state so
    the start-up transient is as small as possible.
    """
    if L <= 0.0:
        return dP / R
    _, q, _ = lsim(([1.0], [L, R]), dP, t, X0=[dP[0] / R * L])
    return q


def dominant_omega(x, dt):
    """Angular frequency [rad/s] of the strongest AC component of x."""
    freqs = np.fft.rfftfreq(len(x), dt)
    spec = np.abs(np.fft.rfft(x - x.mean()))
    return 2.0 * np.pi * freqs[np.argmax(spec)]


def seed_RL(dP, Q, omega):
    """Analytic single-frequency starting guess.

    DC:          R        = mean(dP) / mean(Q)
    Fundamental: |Z(jw)|  = amp(dP) / amp(Q) = sqrt(R^2 + (w*L)^2)
    """
    R0 = dP.mean() / Q.mean()
    z_mag = amplitude(dP) / amplitude(Q)
    wL = np.sqrt(max(z_mag**2 - R0**2, 0.0))
    return R0, max(wL / omega, 1e-6)


def fit_component(name, dP, Q, t, settle_mask, omega):
    """Optimise (R, L) so predicted flow matches clinical mean + amplitude."""
    q_mean_ref = Q[settle_mask].mean()
    q_amp_ref = amplitude(Q[settle_mask])

    def residuals(p):
        q_pred = predict_flow(p[0], p[1], dP, t)[settle_mask]
        return [(q_pred.mean() - q_mean_ref) / q_mean_ref,
                (amplitude(q_pred) - q_amp_ref) / q_amp_ref]

    R0, L0 = seed_RL(dP, Q, omega)
    sol = least_squares(residuals, [R0, L0],
                        bounds=([1e-6, 0.0], [np.inf, np.inf]))
    R, L = sol.x

    q_pred = predict_flow(R, L, dP, t)
    return {
        'name': name,
        'R': float(R), 'L': float(L),
        'R_seed': float(R0), 'L_seed': float(L0),
        'tau_s': float(L / R),
        'dP_mean': float(dP.mean()), 'dP_amp': float(amplitude(dP)),
        'q_pred_mean': float(q_pred[settle_mask].mean()),
        'q_pred_amp': float(amplitude(q_pred[settle_mask])),
        'q_clin_mean': float(q_mean_ref), 'q_clin_amp': float(q_amp_ref),
        '_q_pred': q_pred,
        '_dP': dP,
    }


# ===========================================================================
def get_uv_reference(t_clin):
    """UV pressure bounding the downstream end of the tubing."""
    if isinstance(UV_SOURCE, (int, float)):
        print(f"UV reference: user-supplied constant = {float(UV_SOURCE):.4f} mmHg")
        return np.full_like(t_clin, float(UV_SOURCE)), float(UV_SOURCE)

    print(f"Running AP_model_v12_6 for {SIM_DURATION_S:.0f} s to obtain P_UV ...")
    model = ArtificialPlacentaModel(reflex_on=True)
    sol = model.run_simulation(duration=SIM_DURATION_S)
    p_uv = sol.y[16] / model.params['C_UV']

    steady = sol.t >= SETTLE_S
    uv_mean = float(p_uv[steady].mean())
    print(f"  solver steps: {len(sol.t)}")
    print(f"  P_UV: mean = {uv_mean:.4f} mmHg, "
          f"p{AMP_LO_PCT:.0f}-p{AMP_HI_PCT:.0f} amplitude = "
          f"{amplitude(p_uv[steady]):.4f} mmHg")

    if UV_SOURCE == 'model_mean':
        return np.full_like(t_clin, uv_mean), uv_mean
    if UV_SOURCE == 'model_waveform':
        return np.interp(t_clin, sol.t, p_uv), uv_mean
    raise ValueError(f"Unrecognised UV_SOURCE: {UV_SOURCE!r}")


def summary_plots(results, t, Q):
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21, 9))

    fig.suptitle(
        f"AP component R-L fits  |  clinical window {START_TIME} - {END_TIME} "
        f"({t[-1]:.1f} s)  |  UV source: {UV_SOURCE}",
        fontsize=16, fontweight='bold')

    for col, r in enumerate(results):
        # Top row: driving pressure drop
        ax = axes[0, col]
        ax.plot(t, r['_dP'], color='purple', lw=1.5)
        ax.set_title(f"{r['name']}  ΔP  (mean {r['dP_mean']:.2f}, "
                     f"amp {r['dP_amp']:.2f} mmHg)", fontsize=13)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔP (mmHg)')
        ax.grid(True, alpha=0.3)

        # Bottom row: clinical vs predicted flow
        ax = axes[1, col]
        ax.plot(t, Q * 60.0, color='green', lw=1.5, label='Clinical flow')
        ax.plot(t, r['_q_pred'] * 60.0, color='red', lw=1.5,
                label=f"R-L predicted  (R={r['R']:.4f}, L={r['L']:.4f})")
        ax.axvspan(0, SETTLE_S, color='grey', alpha=0.15)
        ax.axhline(r['q_clin_mean'] * 60.0, color='green', ls=':', lw=1)
        ax.axhline(r['q_pred_mean'] * 60.0, color='red', ls=':', lw=1)
        ax.set_title(f"{r['name']}  flow  |  mean {r['q_pred_mean'] * 60:.1f} vs "
                     f"{r['q_clin_mean'] * 60:.1f}  |  amp "
                     f"{r['q_pred_amp'] * 60:.1f} vs {r['q_clin_amp'] * 60:.1f} mL/min",
                     fontsize=13)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flow (mL/min)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.89)
    plt.show()


# ===========================================================================
if __name__ == '__main__':
    csv_path = clinical_csv()

    # ---- 1. clinical window (same 30 s window used by AP_model_v12_6) ----
    clin = ClinicalData(csv_path)
    mask = clin.get_time_window_indices(START_TIME, END_TIME)
    if mask.sum() == 0:
        raise ValueError(f"No clinical rows found between {START_TIME} and {END_TIME}.")

    t = clin._times[mask] - clin._times[mask][0]
    Q = clin._flow_ml_s[mask]
    p_pre_ecmo = clin._p_pre_ecmo[mask]
    p_post_ecmo = clin._p_post_ecmo[mask]
    p_post_res = clin._p_post_res[mask]

    dt = 1.0 / ClinicalData.SAMPLE_RATE_HZ
    omega = dominant_omega(Q, dt)

    print(f"Clinical window: {START_TIME} -> {END_TIME}  "
          f"({t[-1]:.1f} s, {mask.sum()} samples @ {ClinicalData.SAMPLE_RATE_HZ:.0f} Hz)")
    print(f"Clinical flow  : mean = {Q.mean() * 60:.2f} mL/min, "
          f"amplitude = {amplitude(Q) * 60:.2f} mL/min")
    print(f"Dominant flow frequency: {omega / (2 * np.pi):.3f} Hz "
          f"({omega / (2 * np.pi) * 60:.1f} bpm), omega = {omega:.3f} rad/s\n")

    # ---- 2. UV reference for the tubing ----
    p_uv, uv_mean = get_uv_reference(t)
    print()

    # ---- 3. pressure drop across each passive component ----
    components = [
        ('ECMO',       p_pre_ecmo - p_post_ecmo),
        ('Restrictor', p_post_ecmo - p_post_res),
        ('Tubing',     p_post_res - p_uv),
    ]

    # ---- 4. fit ----
    settle_mask = t >= SETTLE_S
    results = [fit_component(n, dP, Q, t, settle_mask, omega)
               for n, dP in components]

    # ---- 5. report ----
    print("=" * 96)
    print(f"{'Component':<12}{'R [mmHg.s/mL]':>16}{'L [mmHg.s2/mL]':>17}"
          f"{'tau=L/R [s]':>14}{'Q mean err':>13}{'Q amp err':>12}")
    print("-" * 96)
    for r in results:
        print(f"{r['name']:<12}{r['R']:>16.4f}{r['L']:>17.4f}{r['tau_s']:>14.4f}"
              f"{(r['q_pred_mean'] / r['q_clin_mean'] - 1) * 100:>12.3f}%"
              f"{(r['q_pred_amp'] / r['q_clin_amp'] - 1) * 100:>11.3f}%")
    print("-" * 96)
    R_tot = sum(r['R'] for r in results)
    L_tot = sum(r['L'] for r in results)
    print(f"{'TOTAL':<12}{R_tot:>16.4f}{L_tot:>17.4f}{L_tot / R_tot:>14.4f}")
    print("=" * 96)
    print("\nSeries totals above are what set the AP loop dynamics in AP_model_v13.\n")

    for r in results:
        print(f"{r['name']:<12} seed (single-frequency): "
              f"R = {r['R_seed']:.4f}, L = {r['L_seed']:.4f}   ->   "
              f"fitted: R = {r['R']:.4f}, L = {r['L']:.4f}")

    payload = {
        'window': {'start': START_TIME, 'end': END_TIME,
                   'duration_s': float(t[-1]), 'n_samples': int(mask.sum())},
        'uv_source': UV_SOURCE, 'uv_mean_mmHg': uv_mean,
        'units': {'R': 'mmHg*s/mL', 'L': 'mmHg*s^2/mL'},
        'components': {r['name']: {k: v for k, v in r.items()
                                   if not k.startswith('_') and k != 'name'}
                       for r in results},
        'totals': {'R': R_tot, 'L': L_tot},
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")

    summary_plots(results, t, Q)
