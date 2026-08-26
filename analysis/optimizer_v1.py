"""
Parameter optimizer for AP_model_v14.

Optimizes three parameters against the clinical data:
    1. R_AO4_UA      - aorta(AO4) -> umbilical artery resistance   [mmHg*s/mL]
    2. V_BLOOD_TOTAL - total initial blood volume in the model      [mL]
    3. C_UA          - umbilical artery compliance                  [mL/mmHg]

Objective (the "sum" being minimized) = sum of the model-vs-clinical
differences of three features:
    a. mean AP flow                             [mL/min]
    b. AP flow amplitude (peak-to-peak)         [mL/min]
    c. pressure drop, pre-pump -> post-restrictor (mean)  [mmHg]

Optimizer: scipy.optimize.minimize(method='SLSQP'), the closest Python
equivalent of MATLAB's fmincon (bounded, gradient-based SQP).  'trust-constr'
(fmincon's interior-point analogue) is also selectable below.

PERSISTENCE / RESUMING
    The best parameter set found so far is written to `results/optimizer/optimizer_state.json`
    after every single improvement.  Re-running this file picks up from that
    state and keeps refining, so results monotonically improve the longer /
    the more times you run it.  Killing the run with Ctrl-C never loses the
    best-so-far.

USAGE
    python3 analysis/optimizer_v1.py                 # one session (N_RESTARTS starts)
    python3 analysis/optimizer_v1.py --reset         # wipe saved state, start fresh
    python3 analysis/optimizer_v1.py --show          # just print the current best
    python3 analysis/optimizer_v1.py --restarts 10   # override restart count
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Bootstrap: put the project root on sys.path so this file runs directly
# (python3 analysis/optimizer_v1.py) from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models.AP_model_v14 as M
from models.AP_model_v14 import (ArtificialPlacentaModel, ClinicalData,
                                 START_TIME, END_TIME)
from project_paths import RESULTS_OPTIMIZER, clinical_csv, ensure

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
STATE_FILE = os.path.join(ensure(RESULTS_OPTIMIZER), "optimizer_state.json")
CSV_PATH = clinical_csv()

# --- simulation ------------------------------------------------------------
SIM_DURATION = 20.0      # [s] length of each model run
ANALYSIS_SECONDS = 8.0   # [s] only the LAST this-many seconds are scored
                         #     (discards the start-up transient)
# Two-stage solver fidelity.  SLSQP needs ~4 evaluations per iteration, so
# searching at the model's native 0.002 would cost ~5x the wall time.  The cost
# landscape has nearly the same SHAPE at both settings, but NOT the same values:
# at the working parameter set mean_flow reads 174.6 at 0.01 vs 178.7 at 0.002.
# Searching coarse therefore lands slightly off-target, so the winner is
# re-converged at full fidelity in a short polish phase at the end.
MAX_STEP_COARSE = 0.01   # used for the SLSQP search
MAX_STEP_POLISH = 0.002  # matches AP_model_v14/15/16.run_simulation exactly
MAX_STEP = MAX_STEP_COARSE   # mutated at runtime by the polish phase
RTOL, ATOL = 1e-6, 1e-8  # tight tolerances -> smooth objective for the
                         # finite-difference gradients SLSQP needs

# --- objective -------------------------------------------------------------
# True  -> each term is a RELATIVE error (|model-clin| / |clin|), so flow
#          (~188 mL/min) and pressure (~51 mmHg) contribute comparably.
# False -> raw absolute differences are summed in their own units, which lets
#          the largest-magnitude feature (mean flow) dominate.
NORMALIZE_TERMS = True
WEIGHTS = {"mean_flow": 1.0, "flow_amplitude": 1.0, "pressure_drop": 1.0}

# Peak-to-peak is measured between these percentiles so a single noisy clinical
# sample cannot define the amplitude.  Use (0, 100) for a true min/max.
AMP_PCTL = (2.5, 97.5)

# --- optimizer -------------------------------------------------------------
METHOD = "SLSQP"         # or "trust-constr"
N_RESTARTS = 4           # optimizer starts per session (1st = best-so-far)
MAXITER = 60             # SQP iterations per start (coarse phase)
MAXITER_POLISH = 25      # SQP iterations for the full-fidelity polish
FD_STEP = 1e-3           # finite-difference step, in log-space
MAX_EVALS_PER_SESSION = 1500   # hard safety cap on objective evaluations

# --- parameter bounds ------------------------------------------------------
# (min, max) in physical units.  Bounds act as fmincon's lb/ub.
# Centred on the values baked into AP_model_v15/v16, which are the ones that
# actually fit (cost 0.107).  The previous bounds were centred on the v14
# BASELINE (7.0271 / 145.27 / 0.06573), which excluded the working values by
# 2.7x above and 10.6x below -- the optimizer could not reach the answer and
# parked at cost 0.963 with a flow amplitude of 2.1 mL/min against a clinical
# 40.1.  Do not re-centre these on the baseline again.
R_AO4_UA_WORKING      = 40.545318   # = res_AO4_UA in AP_model_v15/v16
V_BLOOD_TOTAL_WORKING = 268.787     # = BLOOD_VOLUME
C_UA_WORKING          = 0.002471    # = cap_UA

BOUNDS = {
    "R_AO4_UA":      (R_AO4_UA_WORKING*0.4,      R_AO4_UA_WORKING*2.1),       # mmHg*s/mL
    "V_BLOOD_TOTAL": (V_BLOOD_TOTAL_WORKING*0.4, V_BLOOD_TOTAL_WORKING*2.1),  # mL
    "C_UA":          (C_UA_WORKING*0.4,          C_UA_WORKING*2.1),           # mL/mmHg
}
PARAM_NAMES = ["R_AO4_UA", "V_BLOOD_TOTAL", "C_UA"]

P_INIT = 16.5            # [mmHg] filling pressure used to distribute volume
                         # (from AP_model_v14.run_simulation)


# ===========================================================================
#  MODEL EVALUATION
# ===========================================================================
def _compliance_list(model):
    """Compliances in the same order as state vector entries 0..18."""
    p = model.params
    return [
        p["C_AR"], p["C_AO1"], p["C_AO2"], p["C_AO3"], p["C_AO4"],
        p["C_CA"], p["C_BR"], p["C_UB"], p["C_PA1"], p["C_PA2"],
        p["C_LUNG"], p["C_KID"], p["C_HE"], p["C_INTE"], p["C_LEG"],
        p["C_UA"], p["C_UV"], p["C_SVC"], p["C_IVC"],
    ]


def baseline_blood_volume():
    """Total initial blood volume implied by the un-tuned AP_model_v14."""
    model = ArtificialPlacentaModel(reflex_on=True)
    v = np.array([c * P_INIT for c in _compliance_list(model)])
    return float(v.sum() + 4.0 + 2.0 + 10.4 + 6.5)


def build_model(R_AO4_UA, C_UA):
    """A v14 model with the two tuned parameters overridden."""
    model = ArtificialPlacentaModel(reflex_on=True)
    model.params["R_AO4_UA"] = float(R_AO4_UA)
    model.params["C_UA"] = float(C_UA)
    return model


def make_x0(model, V_blood_total):
    """
    Initial state, built exactly like AP_model_v14.run_simulation but with
    every compartment volume rescaled so the circuit holds V_blood_total mL.
    The AP flow state starts at its R-only steady value for consistency.
    """
    x0 = np.zeros(25)
    for i, c in enumerate(_compliance_list(model)):
        x0[i] = c * P_INIT
    x0[19], x0[20], x0[21], x0[22] = 4.0, 2.0, 10.4, 6.5

    x0[0:23] *= float(V_blood_total) / x0[0:23].sum()

    x0[23] = 55.0
    P_UA0 = x0[15] / model.params["C_UA"]
    P_UV0 = x0[16] / model.params["C_UV"]
    x0[24] = model.steady_state_flow(P_UA0, P_UV0)
    return x0


def simulate(R_AO4_UA, V_blood_total, C_UA):
    model = build_model(R_AO4_UA, C_UA)
    sol = solve_ivp(
        model.compute_derivatives,
        [0.0, SIM_DURATION],
        make_x0(model, V_blood_total),
        method="Radau",
        max_step=MAX_STEP,
        rtol=RTOL,
        atol=ATOL,
    )
    return sol, model


# The clinical trace is uniformly sampled at 10 Hz, but Radau is adaptive: it
# clusters steps around the systolic upstroke and stretches them through
# diastole.  np.mean/np.percentile weight every stored SAMPLE equally, so on the
# raw solver grid they answer a different question than they do on the clinical
# data (sample-weighted vs time-weighted).  Resampling onto an even grid first
# makes both time-weighted.  Measured bias on the working parameter set: flow
# amplitude 39.94 raw vs 40.54 resampled (1.5%).
RESAMPLE_HZ = 200.0      # uniform grid for model metrics; >> the 10 Hz clinical rate


def model_metrics(sol, model):
    """Mean AP flow [mL/min], flow amplitude [mL/min], pre-pump -> post-restrictor drop [mmHg]."""
    t_cut = sol.t[-1] - ANALYSIS_SECONDS
    sel = sol.t >= t_cut
    if sel.sum() < 10:
        sel = np.ones_like(sol.t, dtype=bool)

    t = sol.t[sel]
    q_raw = sol.y[24, sel] * 60.0                   # mL/min
    P_UA_raw = sol.y[15, sel] / model.params["C_UA"]
    P_UV_raw = sol.y[16, sel] / model.params["C_UV"]

    # Uniform time grid, then every statistic below is a true time-average.
    n_uniform = max(64, int(round((t[-1] - t[0]) * RESAMPLE_HZ)))
    t_u = np.linspace(t[0], t[-1], n_uniform)
    q = np.interp(t_u, t, q_raw)
    P_UA = np.interp(t_u, t, P_UA_raw)
    P_UV = np.interp(t_u, t, P_UV_raw)

    dQ, pump_gain, pre_ecmo, post_ecmo, post_res = model.ap_loop(
        P_UA, P_UV, q / 60.0
    )

    return {
        "mean_flow": float(q.mean()),
        "flow_amplitude": float(
            np.percentile(q, AMP_PCTL[1]) - np.percentile(q, AMP_PCTL[0])
        ),
        "pressure_drop": float(np.mean(P_UA - post_res)),
    }


def clinical_metrics(clin):
    mask = clin.get_time_window_indices(START_TIME, END_TIME)
    if mask.sum() == 0:
        raise ValueError(f"No clinical rows between {START_TIME} and {END_TIME}.")

    q = clin._flow_ml_s[mask] * 60.0                # mL/min
    dp = clin._p_pre_pump[mask] - clin._p_post_res[mask]

    return {
        "mean_flow": float(q.mean()),
        "flow_amplitude": float(
            np.percentile(q, AMP_PCTL[1]) - np.percentile(q, AMP_PCTL[0])
        ),
        "pressure_drop": float(dp.mean()),
    }


def cost_from_metrics(mod_m, clin_m):
    """The scalar 'sum' that fmincon/SLSQP minimizes, plus its breakdown."""
    terms = {}
    for k in ("mean_flow", "flow_amplitude", "pressure_drop"):
        diff = abs(mod_m[k] - clin_m[k])
        if NORMALIZE_TERMS:
            diff /= max(abs(clin_m[k]), 1e-9)
        terms[k] = WEIGHTS[k] * diff
    return float(sum(terms.values())), terms


# ===========================================================================
#  PERSISTENT STATE
# ===========================================================================
def objective_signature():
    """Costs are only comparable across runs if the objective is unchanged."""
    return {
        "sim_duration": SIM_DURATION,
        "analysis_seconds": ANALYSIS_SECONDS,
        "max_step_coarse": MAX_STEP_COARSE,
        "max_step_polish": MAX_STEP_POLISH,
        "resample_hz": RESAMPLE_HZ,
        "normalize": NORMALIZE_TERMS,
        "weights": WEIGHTS,
        "amp_pctl": list(AMP_PCTL),
        "window": [START_TIME, END_TIME],
        "params": PARAM_NAMES,
    }


def load_state():
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] could not read {STATE_FILE}: {e}  -- starting fresh.")
        return None

    if state.get("signature") != objective_signature():
        print("[warn] objective settings changed since the saved state was written.")
        print("       Old best params will be reused as a starting point, but the")
        print("       saved cost is not comparable and will be re-evaluated.")
        state["best"]["cost"] = None
    return state


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)   # atomic: never leaves a half-written file


def new_state():
    return {
        "signature": objective_signature(),
        "best": {"params": None, "cost": None, "metrics": None},
        "clinical": None,
        "n_sessions": 0,
        "n_evals_total": 0,
        "history": [],
    }


# ===========================================================================
#  LOG-SPACE SCALING  (all three params become O(1) -> well-scaled gradients)
# ===========================================================================
def to_z(params_dict):
    return np.array([np.log(params_dict[n]) for n in PARAM_NAMES])


def to_params(z):
    return {n: float(np.exp(v)) for n, v in zip(PARAM_NAMES, z)}


Z_BOUNDS = [(np.log(BOUNDS[n][0]), np.log(BOUNDS[n][1])) for n in PARAM_NAMES]


def clip_z(z):
    return np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(z, Z_BOUNDS)])


# ===========================================================================
#  OPTIMIZATION DRIVER
# ===========================================================================
class Objective:
    """Callable objective that caches, logs, and saves every new best."""

    def __init__(self, clin_m, state):
        self.clin_m = clin_m
        self.state = state
        self.cache = {}
        self.n_evals = 0
        self.t0 = time.time()

    def evaluate(self, params):
        """Full evaluation of a physical parameter dict -> (cost, metrics, terms)."""
        key = tuple(round(params[n], 10) for n in PARAM_NAMES)
        if key in self.cache:
            return self.cache[key]

        try:
            sol, model = simulate(
                params["R_AO4_UA"], params["V_BLOOD_TOTAL"], params["C_UA"]
            )
            if not sol.success or sol.t[-1] < SIM_DURATION * 0.99:
                raise RuntimeError(sol.message)
            mod_m = model_metrics(sol, model)
            if not np.all(np.isfinite(list(mod_m.values()))):
                raise RuntimeError("non-finite model metrics")
            cost, terms = cost_from_metrics(mod_m, self.clin_m)
        except Exception as e:
            # A failed/stiff integration must not kill the session -- return a
            # large finite penalty so the optimizer simply walks away from it.
            print(f"    [sim failed: {e}]")
            cost, mod_m, terms = 1e6, None, None

        result = (cost, mod_m, terms)
        self.cache[key] = result
        return result

    def __call__(self, z):
        if self.n_evals >= MAX_EVALS_PER_SESSION:
            raise StopIteration("evaluation budget exhausted")

        params = to_params(z)
        cost, mod_m, terms = self.evaluate(params)
        self.n_evals += 1
        self.state["n_evals_total"] += 1

        best = self.state["best"]
        if mod_m is not None and (best["cost"] is None or cost < best["cost"] - 1e-12):
            best["params"] = params
            best["cost"] = cost
            best["metrics"] = mod_m
            best["terms"] = terms
            best["timestamp"] = datetime.now().isoformat(timespec="seconds")
            best["eval_index"] = self.state["n_evals_total"]
            self.state["history"].append(copy.deepcopy(best))
            save_state(self.state)          # <-- improvement is durable NOW
            print(
                f"  [{self.n_evals:4d}] NEW BEST cost={cost:.6f}   "
                f"R_AO4_UA={params['R_AO4_UA']:.4f}  "
                f"V_blood={params['V_BLOOD_TOTAL']:.2f}  "
                f"C_UA={params['C_UA']:.5f}   "
                f"({time.time() - self.t0:.0f}s)"
            )
        return cost


def run_optimization(obj, z_start, tag, maxiter=None):
    maxiter = MAXITER if maxiter is None else maxiter
    print(f"\n--- start {tag}: {fmt_params(to_params(z_start))}")
    opts = (
        {"maxiter": maxiter, "ftol": 1e-9, "eps": FD_STEP, "disp": False}
        if METHOD == "SLSQP"
        else {"maxiter": maxiter, "finite_diff_rel_step": FD_STEP, "verbose": 0}
    )
    try:
        res = minimize(obj, z_start, method=METHOD, bounds=Z_BOUNDS, options=opts)
        print(f"    finished: {res.message}  (f={res.fun:.6f}, {res.nit} iters)")
        return True
    except StopIteration as e:
        print(f"    stopped: {e}")
        return False


def fmt_params(p):
    return (
        f"R_AO4_UA={p['R_AO4_UA']:.4f}  "
        f"V_BLOOD_TOTAL={p['V_BLOOD_TOTAL']:.2f}  "
        f"C_UA={p['C_UA']:.5f}"
    )


def print_summary(state, clin_m):
    best = state["best"]
    print("\n" + "=" * 72)
    print("BEST PARAMETERS SO FAR")
    print("=" * 72)
    if best["params"] is None:
        print("  (none yet)")
        return

    p = best["params"]
    cost_str = ("not scored under the current settings"
                if best["cost"] is None else f"{best['cost']:.6f}")
    print(f"  objective sum : {cost_str}"
          f"   ({'normalized' if NORMALIZE_TERMS else 'raw units'})")
    print(f"  sessions run  : {state['n_sessions']}   "
          f"total evaluations: {state['n_evals_total']}")
    print()
    fidelity = best.get("max_step")
    if fidelity is not None:
        print(f"  scored at     : max_step={fidelity}"
              f"{'  (full fidelity)' if fidelity == MAX_STEP_POLISH else '  (COARSE - not polished)'}")
    print()
    print("  Paste these into the constant block at the top of "
          "models/AP_model_v15.py and models/AP_model_v16.py:")
    print(f"      res_AO4_UA   = {p['R_AO4_UA']:.6f}")
    print(f"      BLOOD_VOLUME = {p['V_BLOOD_TOTAL']:.6f}")
    print(f"      cap_UA       = {p['C_UA']:.6f}")
    print(f"      (current values there: 40.545318 / 268.787 / 0.002471)")
    print(f"      NOTE v15 scales x0 by BLOOD_VOLUME/145.229, where 145.229 is the")
    print(f"      un-scaled baseline total -- that literal stays as it is.")
    print()
    if best.get("metrics") is None:
        return
    print(f"  {'feature':<18}{'clinical':>14}{'model':>14}{'diff':>14}{'term':>12}")
    for k in ("mean_flow", "flow_amplitude", "pressure_drop"):
        unit = "mmHg" if k == "pressure_drop" else "mL/min"
        print(f"  {k + ' (' + unit + ')':<18}"
              f"{clin_m[k]:>14.4f}{best['metrics'][k]:>14.4f}"
              f"{best['metrics'][k] - clin_m[k]:>14.4f}"
              f"{best.get('terms', {}).get(k, float('nan')):>12.5f}")
    print("=" * 72)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="discard saved state")
    ap.add_argument("--show", action="store_true", help="print best and exit")
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--no-polish", action="store_true",
                    help="skip the full-fidelity polish phase at the end")
    args = ap.parse_args()

    if args.reset and os.path.isfile(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"Removed {STATE_FILE}")

    clin = ClinicalData(CSV_PATH)
    clin_m = clinical_metrics(clin)

    state = load_state() or new_state()
    state["clinical"] = clin_m
    state["signature"] = objective_signature()

    if args.show:
        print_summary(state, clin_m)
        return

    print(f"Clinical window {START_TIME} -> {END_TIME}")
    print(f"Clinical targets: mean_flow={clin_m['mean_flow']:.3f} mL/min, "
          f"amplitude={clin_m['flow_amplitude']:.3f} mL/min, "
          f"dP(pre-pump->post-restrictor)={clin_m['pressure_drop']:.3f} mmHg")

    obj = Objective(clin_m, state)

    # -- starting points ----------------------------------------------------
    if state["best"]["params"] is not None:
        z_best = clip_z(to_z(state["best"]["params"]))
        if state["best"]["cost"] is None:      # objective changed -> re-score
            obj(z_best)
        c = state["best"]["cost"]
        print("Resuming from saved best "
              + (f"(cost={c:.6f})" if c is not None else "(cost not yet scored)"))
    else:
        # Start from the values baked into v15/v16.  The v14 BASELINE
        # (7.027 / 145.229 / 0.06573) now sits outside BOUNDS and would simply
        # be clipped to the box edge, throwing away a known-good starting fit.
        z_best = clip_z(to_z({
            "R_AO4_UA": R_AO4_UA_WORKING,
            "V_BLOOD_TOTAL": V_BLOOD_TOTAL_WORKING,
            "C_UA": C_UA_WORKING,
        }))
        print("No saved state -- starting from the AP_model_v15/v16 working values.")

    state["n_sessions"] += 1
    # Perturbations shrink as more sessions accumulate: early sessions explore,
    # later ones polish around the incumbent.
    rng = np.random.default_rng()
    spread = 0.45 / (1.0 + 0.35 * state["n_sessions"])

    try:
        for i in range(max(1, args.restarts)):
            if i == 0:
                z0 = clip_z(to_z(state["best"]["params"])
                            if state["best"]["params"] else z_best)
                tag = "restart 1/%d (incumbent best)" % args.restarts
            else:
                anchor = to_z(state["best"]["params"])
                z0 = clip_z(anchor + rng.normal(0.0, spread, size=len(PARAM_NAMES)))
                tag = f"restart {i + 1}/{args.restarts} (perturbed, sigma={spread:.3f})"

            if not run_optimization(obj, z0, tag):
                break
    except KeyboardInterrupt:
        print("\nInterrupted -- best-so-far is already saved.")

    # ---- polish: re-converge the winner at the model's native fidelity ----
    # Costs from the coarse phase are NOT comparable to fine-fidelity costs, so
    # this runs with a fresh Objective (fresh cache) and re-scores the incumbent
    # before comparing anything.
    if not args.no_polish and state["best"]["params"] is not None:
        global MAX_STEP
        MAX_STEP = MAX_STEP_POLISH
        print(f"\n{'=' * 72}\nPOLISH PHASE  --  re-converging at max_step={MAX_STEP_POLISH} "
              f"(model fidelity)\n{'=' * 72}")
        coarse = copy.deepcopy(state["best"])
        state["best_coarse"] = coarse
        state["best"]["cost"] = None      # force a re-score at the new fidelity
        obj_fine = Objective(clin_m, state)
        z_in = clip_z(to_z(state["best"]["params"]))
        c_fine = obj_fine(z_in)
        print(f"  incumbent re-scored at full fidelity: "
              f"{coarse['cost']:.6f} (coarse) -> {c_fine:.6f} (fine)")
        try:
            run_optimization(obj_fine, z_in, f"polish (maxiter={MAXITER_POLISH})",
                             maxiter=MAXITER_POLISH)
        except KeyboardInterrupt:
            print("\nPolish interrupted -- best-so-far is already saved.")
        state["best"]["max_step"] = MAX_STEP_POLISH
    elif state["best"]["params"] is not None:
        state["best"]["max_step"] = MAX_STEP_COARSE

    save_state(state)
    print(f"\nState written to {STATE_FILE}  "
          f"({obj.n_evals} evaluations this session, "
          f"{time.time() - obj.t0:.0f}s)")
    print_summary(state, clin_m)


if __name__ == "__main__":
    main()
