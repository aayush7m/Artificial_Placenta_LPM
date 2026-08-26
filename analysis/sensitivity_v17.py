"""
Sensitivity analysis and re-fit of AP_model_v17's identifiable mechanism
parameters.  Consumes the output of analysis/profile_likelihood_v17.py and
produces the parameter set that AP_model_v18 is built from.

--------------------------------------------------------------------------
WHY THIS RUNS SECOND
--------------------------------------------------------------------------
The profile likelihood answers "which parameters CAN the data constrain".
Only those should be fitted -- optimising a structurally non-identifiable
parameter moves it to an arbitrary value with no improvement in cost and no
physiological justification, and it will not reproduce on a different dataset.
So this script takes the top TOP_N parameters ranked by profile depth
(max dJ over the parameter's range), restricted to those the profile called
identifiable or practically identifiable, and fits only those.

--------------------------------------------------------------------------
WHAT IT COMPUTES
--------------------------------------------------------------------------
1. LOCAL SENSITIVITY MATRIX  S[f, i] = (d f_f / d theta_i) * (theta_i / sigma_f)
   Central finite differences at the reference point, in relative parameter
   units and clinical-sigma feature units.  S[f,i] is therefore "how many
   sigmas of feature f does a 1% change in parameter i buy" -- directly
   comparable across every parameter/feature pair regardless of units.

2. COLLINEARITY.  The correlation matrix of S's columns.  Two parameters with
   |corr| near 1 push the model in the same direction and cannot be fitted
   separately; this is the pairwise companion to the profile likelihood's
   one-at-a-time verdict, and it explains WHY a parameter came back
   non-identifiable when it did.

3. JOINT RE-FIT.  Multi-start SLSQP over the selected mechanism parameters
   plus the 3 circuit nuisance parameters, in log/linear space, against the
   same chi-square cost (analysis/ap_cost.py).  The result is written to
   results/sensitivity_v17/fitted_parameters.json and is what v18 uses.

4. PLOTS (one grid each):
   - sensitivity heatmap, parameters x features
   - tornado chart of total sensitivity per parameter
   - collinearity matrix
   - per-feature cost before vs after the fit
   - simulated vs clinical traces, v17 baseline vs fitted

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python3 analysis/sensitivity_v17.py                # top 6, full fit
    python3 analysis/sensitivity_v17.py --top 10
    python3 analysis/sensitivity_v17.py --quick        # fewer restarts
    python3 analysis/sensitivity_v17.py --plot-only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import ap_cost as A
from analysis import profile_likelihood_v17 as PL
from project_paths import RESULTS_SENSITIVITY_V17, ensure

OUT_DIR    = ensure(RESULTS_SENSITIVITY_V17)
FIT_FILE   = os.path.join(OUT_DIR, "fitted_parameters.json")
PLOT_SENS  = os.path.join(OUT_DIR, "sensitivity_grid.png")
PLOT_FIT   = os.path.join(OUT_DIR, "fit_comparison.png")

TOP_N       = 6      # mechanism parameters carried into the joint fit
# Sized to a predictable ~30 min on 6 performance cores.  Each SLSQP iteration
# costs (n_params + 1) evaluations for the finite-difference gradient, so with
# 9 free parameters a restart is ~10 evaluations per iteration; at ~5 s each the
# defaults this replaced (6 restarts x 120 iterations) would have run ~100 min.
# 4 restarts fit the 6-worker pool in a single round, and the first start is the
# reference point, which is already close.
N_RESTARTS  = 4
MAXITER     = 40
FD_REL      = 0.02   # +/-2% relative perturbation for the sensitivity matrix
COMPARE_SECONDS = 4.0  # [s] window drawn in the trace panels.  The scored
                       # window is 8 s and the clinical recording is 31 s, but
                       # at ~140 bpm that is 70 overlapping beats and renders as
                       # a solid block.  4 s shows the waveform shape, which is
                       # what the comparison is actually about.
MAX_STEP    = A.MAX_STEP_FINE
N_WORKERS   = max(1, min(os.cpu_count() or 4, 10) - 1)

# Verdicts worth fitting.  "structural" parameters are excluded by
# construction: the data cannot distinguish any value of them from any other.
FITTABLE = ("identifiable", "practical")


# ===========================================================================
#  SENSITIVITY MATRIX
# ===========================================================================
def _feat_task(args):
    overrides, = args
    with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
        sol, model = A.simulate(overrides, max_step=MAX_STEP,
                                duration=A.SIM_DURATION)
    if sol is None:
        return None
    try:
        return A.model_features(sol, model)
    except Exception:
        return None


def sensitivity_matrix(base, names, pool):
    """S[f, i] = (d f_f / d ln theta_i) / sigma_f, by central differences.

    Differentiating with respect to ln(theta) rather than theta makes every
    column a response to the same RELATIVE perturbation, so parameters whose
    magnitudes span 1e-7 to 7e4 become comparable without any hand-chosen
    scaling.  Dividing by the clinical sigma_f does the same for the rows.
    """
    c_f, s_f, _, _ = A.clinical_targets()

    jobs, meta = [], []
    for i, nm in enumerate(names):
        p = A.MECH_BY_NAME.get(nm)
        v0 = base[nm]
        if p is not None and p.scale == "linear":
            # A linear-scale parameter may legitimately sit at 0 (e.g. a gain
            # switched off), where a relative step is undefined.  Step by a
            # fraction of the parameter's own range instead, and report the
            # derivative per that same step so the columns stay comparable.
            h = FD_REL * (p.hi - p.lo)
            lo_v, hi_v = max(p.lo, v0 - h), min(p.hi, v0 + h)
        else:
            lo_v, hi_v = v0 * (1 - FD_REL), v0 * (1 + FD_REL)
        jobs.append(({**base, nm: lo_v},))
        jobs.append(({**base, nm: hi_v},))
        meta.append((nm, lo_v, hi_v, v0))

    results = pool.map(_feat_task, jobs)

    S = np.zeros((len(A.FEATURES), len(names)))
    for i, (nm, lo_v, hi_v, v0) in enumerate(meta):
        f_lo, f_hi = results[2 * i], results[2 * i + 1]
        if f_lo is None or f_hi is None:
            S[:, i] = np.nan
            continue
        for j, feat in enumerate(A.FEATURES):
            d = (f_hi[feat] - f_lo[feat]) / s_f[feat]
            # normalise by the fractional size of the step actually taken, so
            # every column is "sigmas of feature per 1% of the parameter"
            frac = (hi_v - lo_v) / max(abs(v0), 1e-12) if abs(v0) > 1e-12 \
                else (hi_v - lo_v)
            S[j, i] = d / max(frac, 1e-12) * 0.01
    return S


def collinearity(S):
    """Correlation matrix of S's columns; |corr| ~ 1 means the two parameters
    are indistinguishable to this dataset (a flat valley in the landscape)."""
    X = np.nan_to_num(S.copy())
    norms = np.linalg.norm(X, axis=0)
    norms[norms < 1e-15] = 1.0
    Xn = X / norms
    return Xn.T @ Xn


# ===========================================================================
#  JOINT FIT
# ===========================================================================
def _fit_start(args):
    """One SLSQP restart.  Runs in a worker."""
    names, z0, base, bounds, maxiter = args

    def obj(z):
        ov = dict(base)
        for nm, zi in zip(names, z):
            ov[nm] = PL.from_z(nm, zi)
        return A.evaluate(ov, max_step=MAX_STEP, duration=A.SIM_DURATION)

    res = minimize(obj, z0, method="SLSQP", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-7, "eps": 1e-3})
    vals = {nm: PL.from_z(nm, zi) for nm, zi in zip(names, res.x)}
    return float(res.fun), vals


def joint_fit(base, names, pool, n_restarts=N_RESTARTS, maxiter=MAXITER):
    bounds = [PL.z_bounds(nm) for nm in names]
    rng = np.random.default_rng(20260819)

    starts = [[PL.to_z(nm, base[nm]) for nm in names]]   # start at the reference
    for _ in range(n_restarts - 1):
        z = []
        for nm in names:
            lo, hi = PL.z_bounds(nm)
            z0 = PL.to_z(nm, base[nm])
            # perturb within 25% of the box width, clipped to the box
            z.append(float(np.clip(z0 + rng.normal(0, 0.25 * (hi - lo)), lo, hi)))
        starts.append(z)

    jobs = [(names, z0, base, bounds, maxiter) for z0 in starts]
    best, best_cost = None, np.inf
    t0 = time.time()
    for i, (cost, vals) in enumerate(pool.imap_unordered(_fit_start, jobs), 1):
        print(f"  restart {i}/{len(jobs)}: J = {cost:10.4f}   "
              f"[{(time.time()-t0)/60:.1f} min]")
        if cost < best_cost:
            best, best_cost = vals, cost
    return best, best_cost


# ===========================================================================
#  PLOTS
# ===========================================================================
def plot_sensitivity_grid(S, names, corr, rows, path=PLOT_SENS):
    labels = [A.MECH_BY_NAME[n].label if n in A.MECH_BY_NAME else n for n in names]
    fig = plt.figure(figsize=(19, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.2], wspace=0.42)

    # --- heatmap: features x parameters ---
    ax = fig.add_subplot(gs[0, 0])
    vmax = np.nanmax(np.abs(S)) or 1.0
    im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(A.FEATURES)))
    ax.set_yticklabels(A.FEATURES, fontsize=8)
    ax.set_title("Local sensitivity\n"
                 r"$\sigma$ of feature per 1% change in parameter",
                 fontsize=10, fontweight="bold")
    for j in range(S.shape[0]):
        for i in range(S.shape[1]):
            if abs(S[j, i]) > 0.25 * vmax:
                ax.text(i, j, f"{S[j,i]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="w" if abs(S[j, i]) > 0.6 * vmax else "k")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    # --- tornado: total sensitivity per parameter ---
    ax = fig.add_subplot(gs[0, 1])
    total = np.sqrt(np.nansum(S ** 2, axis=0))
    order = np.argsort(total)
    y = np.arange(len(names))
    ax.barh(y, total[order],
            color=[PL.MECH_COLORS[A.MECH_BY_NAME[names[i]].mech]
                   if names[i] in A.MECH_BY_NAME else "0.5" for i in order])
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax.set_xlabel(r"$\|S_{\cdot i}\|_2$  (total $\sigma$ per 1%)")
    ax.set_title("Overall influence on the fit", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # --- collinearity ---
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(corr, cmap="PuOr", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=8)
    for j in range(len(names)):
        for i in range(len(names)):
            ax.text(i, j, f"{corr[j,i]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="w" if abs(corr[j, i]) > 0.6 else "k")
    ax.set_title("Collinearity\n|r| near 1 = not separately identifiable",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle("Sensitivity analysis -- AP_model_v17 mechanism parameters "
                 "selected by profile likelihood",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_fit_comparison(base, fitted, names, path=PLOT_FIT):
    """Cost breakdown and traces, v17 baseline vs circuit-only vs full fit."""
    import contextlib, io
    c_f, s_f, _, _ = A.clinical_targets()
    clin = A.ClinicalData(A.clinical_csv())
    mask = clin.get_time_window_indices(A.START_TIME, A.END_TIME)
    t_c_full = clin._times[mask] - clin._times[mask][0]
    # Both traces are drawn over the SAME number of seconds, re-zeroed, so the
    # model and the recording are visually comparable rather than one being a
    # short stub next to a long one.  They are not phase-aligned (there is no
    # gating signal), so only waveform shape and amplitude are meaningful here.
    keep_c = t_c_full <= COMPARE_SECONDS
    t_c = t_c_full[keep_c]

    def clip_c(sig):
        return sig[keep_c]

    cases = [
        ("v17 as written", A.v17_circuit_defaults(), "#7f7f7f"),
        ("circuit re-fit only", base, "#1f77b4"),
        ("+ mechanism fit", {**base, **fitted}, "#d62728"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 9))

    traces = []
    for label, ov, col in cases:
        with contextlib.redirect_stdout(io.StringIO()):
            sol, model = A.simulate(ov, max_step=MAX_STEP, duration=A.SIM_DURATION)
        if sol is None:
            traces.append((label, col, None, None, None, None))
            continue
        m_f = A.model_features(sol, model)
        cost, terms = A.cost_from_features(m_f, c_f, s_f)
        sel = sol.t >= sol.t[-1] - A.ANALYSIS_SECONDS
        t = sol.t[sel] - sol.t[sel][0]
        keep_m = t <= COMPARE_SECONDS
        sel = np.flatnonzero(sel)[keep_m]
        t = t[keep_m]
        q = sol.y[A.I_Q_AP, sel] * 60.0
        P_UA = sol.y[A.I_UA, sel] / model.C['UA']
        P_UV = sol.y[A.I_UV, sel] / model.C['UV']
        _, pump, pre_e, post_e, post_r = model.ap_loop(P_UA, P_UV, q / 60.0)
        traces.append((label, col, t, q, (pre_e, post_e, post_r, P_UA),
                       (cost, terms)))

    # --- per-feature cost bars ---
    ax = axes[0, 0]
    w = 0.26
    x = np.arange(len(A.FEATURES))
    for k, (label, col, t, q, P, ct) in enumerate(traces):
        if ct is None:
            continue
        ax.bar(x + (k - 1) * w, [ct[1][f] for f in A.FEATURES], w,
               label=f"{label}  (J={ct[0]:.1f})", color=col)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(A.FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$((m-c)/\sigma)^2$")
    ax.set_title("Cost by feature", fontweight="bold")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # --- AP flow ---
    ax = axes[0, 1]
    ax.plot(t_c, clip_c(clin._flow_ml_s[mask] * 60.0), "g", lw=1.4,
            label="clinical")
    for label, col, t, q, P, ct in traces:
        if t is not None:
            ax.plot(t, q, color=col, lw=1.2, label=label)
    ax.set_xlabel("t (s)"); ax.set_ylabel("mL/min")
    ax.set_title("AP flow", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
              ncol=2).set_zorder(50)
    ax.grid(alpha=0.3)

    # --- pressures ---
    for ax, idx, name, clin_sig in [
        (axes[0, 2], 0, "Pre-ECMO P",        clip_c(clin._p_pre_ecmo[mask])),
        (axes[1, 0], 1, "Post-ECMO P",       clip_c(clin._p_post_ecmo[mask])),
        (axes[1, 1], 2, "Post-restrictor P", clip_c(clin._p_post_res[mask])),
        (axes[1, 2], 3, "Pre-pump (UA) P",   clip_c(clin._p_pre_pump[mask])),
    ]:
        ax.plot(t_c, clin_sig, "g", lw=1.4, label="clinical")
        for label, col, t, q, P, ct in traces:
            if t is not None:
                ax.plot(t, P[idx], color=col, lw=1.2, label=label)
        ax.set_xlabel("t (s)"); ax.set_ylabel("mmHg")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
                  ncol=2).set_zorder(50)
        ax.grid(alpha=0.3)

    fitted_str = ", ".join(f"{k}={v:.4g}" for k, v in list(fitted.items())[:6])
    fig.suptitle(f"AP_model_v17 fit -- {len(names)} mechanism parameters "
                 f"+ 3 circuit parameters\n{fitted_str}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    args = ap.parse_args()

    n_restarts, maxiter = (2, 30) if args.quick else (N_RESTARTS, MAXITER)

    state = PL.load_state()
    if not state.get("scan"):
        print("No profile likelihood state found.\n"
              "Run: python3 analysis/profile_likelihood_v17.py")
        return

    rows = PL.summary_rows(state)
    base = dict(state["ref"])
    circ = state["circuit"]

    # --- pick the parameters to fit ---
    #
    # ONLY profiled parameters are eligible.  A scan depth and a profile depth
    # are not comparable: the scan is a pointwise upper bound, so a parameter
    # reported from its scan keeps an un-collapsed depth while a profiled one
    # has had its nuisance compensation removed.  Ranking the two together puts
    # un-profiled parameters artificially on top -- in this dataset three of
    # them displaced genuinely profiled parameters from the top 6.  Restricting
    # to profiled entries compares like with like.
    eligible = [r for r in rows if r["profiled"] and r["verdict"] in FITTABLE]
    sel = eligible[:args.top]
    n_scan_only = sum(1 for r in rows
                      if not r["profiled"] and r["verdict"] in FITTABLE)
    if n_scan_only:
        print(f"[note] {n_scan_only} parameters were fittable by their scan but "
              f"never profiled; excluded from selection (scan depth is an upper "
              f"bound and is not comparable to a profile depth).")
    names = [r["name"] for r in sel]
    if not names:
        print("Profile likelihood found no identifiable mechanism parameters.\n"
              "Nothing to fit beyond the circuit constants; v18 should use "
              "v17's literature values for every mechanism parameter.")
        names = []

    print(f"J at the reference point (circuit re-fit, mechanisms at v17 "
          f"defaults) = {state['J_ref']:.4f}")
    print(f"\nTop {len(names)} fittable mechanism parameters by profile depth:")
    for i, r in enumerate(sel, 1):
        lo = "-inf" if r["ci_lo"] is None else f"{r['ci_lo']:.4g}"
        hi = "+inf" if r["ci_hi"] is None else f"{r['ci_hi']:.4g}"
        ci = f"[{lo}, {hi}]"
        print(f"  {i}. {r['name']:<22} {r['mech']:<15} dJ={r['span']:9.3f}  "
              f"{r['verdict']:<13} ref={r['ref']:.5g}  95% CI {ci}")

    fit_names = names + A.CIRCUIT_PARAMS

    if args.plot_only and os.path.isfile(FIT_FILE):
        with open(FIT_FILE) as f:
            saved = json.load(f)
        S = np.array(saved["sensitivity_matrix"])
        plot_sensitivity_grid(S, saved["sensitivity_names"],
                              collinearity(S), rows)
        plot_fit_comparison(base, saved["fitted"], saved["fitted_names"])
        return

    with Pool(args.workers) as pool:
        print(f"\n=== Local sensitivity matrix "
              f"({len(fit_names)} params x {len(A.FEATURES)} features) ===")
        S = sensitivity_matrix(base, fit_names, pool)
        corr = collinearity(S)
        total = np.sqrt(np.nansum(S ** 2, axis=0))
        for nm, tt in sorted(zip(fit_names, total), key=lambda kv: -kv[1]):
            print(f"  {nm:<22} |S| = {tt:8.4f}")

        worst = None
        for i in range(len(fit_names)):
            for j in range(i + 1, len(fit_names)):
                if worst is None or abs(corr[i, j]) > abs(corr[worst[0], worst[1]]):
                    worst = (i, j)
        if worst:
            print(f"  most collinear pair: {fit_names[worst[0]]} / "
                  f"{fit_names[worst[1]]}  r = {corr[worst]:.3f}")

        print(f"\n=== Joint fit: {len(names)} mechanism + 3 circuit parameters ===")
        fitted, cost = joint_fit(base, fit_names, pool,
                                 n_restarts=n_restarts, maxiter=maxiter)

    J0 = state["J_ref"]
    print(f"\nJ: {J0:.4f}  ->  {cost:.4f}   "
          f"({100*(1-cost/max(J0,1e-12)):.1f}% reduction)")
    print("\nFitted values:")
    for nm in fit_names:
        print(f"  {nm:<22} {base[nm]:>14.6g}  ->  {fitted[nm]:>14.6g}")

    c_f, s_f, _, _ = A.clinical_targets()
    _, terms = A.evaluate({**base, **fitted}, max_step=MAX_STEP, want_terms=True)
    print(f"\n{'feature':>14}{'clinical':>12}{'sigma':>10}{'cost':>12}")
    for k in A.FEATURES:
        print(f"{k:>14}{c_f[k]:>12.3f}{s_f[k]:>10.3f}{terms[k]:>12.3f}")

    # Provenance the v18 header quotes: what v17 scored as written, and the
    # headline mean-flow mismatch that motivated the whole re-fit.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        sol17, m17 = A.simulate(A.v17_circuit_defaults(), max_step=MAX_STEP,
                                duration=A.SIM_DURATION)
    q_v17 = A.model_features(sol17, m17)["q_mean"] if sol17 is not None else float("nan")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": "AP_model_v17",
        "cost_before": J0,
        "cost_after": cost,
        "cost_v17_as_written": A.evaluate(A.v17_circuit_defaults(),
                                          max_step=MAX_STEP),
        "q_v17": q_v17,
        "q_clinical": c_f["q_mean"],
        "circuit_reference": circ,
        "fitted_names": fit_names,
        "mechanism_names": names,
        "fitted": fitted,
        "reference": {nm: base[nm] for nm in fit_names},
        "sensitivity_names": fit_names,
        "sensitivity_matrix": S.tolist(),
        "features": A.FEATURES,
        "objective": PL.signature(),
        "selection": [{k: r[k] for k in
                       ("name", "mech", "verdict", "span", "ci_lo", "ci_hi")}
                      for r in sel],
    }
    with open(FIT_FILE, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n  wrote {FIT_FILE}")

    print("\n=== Plots ===")
    plot_sensitivity_grid(S, fit_names, corr, rows)
    plot_fit_comparison(base, fitted, fit_names)
    print("\nNext: python3 models/AP_model_v18.py")


if __name__ == "__main__":
    main()
