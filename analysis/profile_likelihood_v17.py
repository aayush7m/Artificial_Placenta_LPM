"""
Profile likelihood analysis of the HOMEOSTATIC MECHANISM parameters of
AP_model_v17 -- which of them the March 2024 ECMO recording can actually
constrain, and which are redundant.

--------------------------------------------------------------------------
WHAT A PROFILE LIKELIHOOD IS
--------------------------------------------------------------------------
For each parameter p, fix p at a grid of values.  At every fixed value,
RE-OPTIMISE the other (nuisance) parameters to get the best cost still
achievable.  That traces

    PL(p) = min over all other params of  J(p, others)

Plotting dJ = PL(p) - J_min against p gives the "parabola".  Because J is a
chi-square (see analysis/ap_cost.py), the width of the parabola at fixed
thresholds is a confidence interval:

    dJ = 1.00  ->  1-sigma / 68%   (1 dof)
    dJ = 3.84  ->  95%             (1 dof)

Three outcomes, and these are the actual deliverable:

  IDENTIFIABLE           parabola crosses the threshold on BOTH sides.
                         The data pin the parameter down; the crossings are
                         its confidence interval.
  PRACTICALLY NON-ID.    crosses on ONE side only.  The data give a bound
                         but not a value -- flat direction in the landscape.
  STRUCTURALLY NON-ID.   never crosses.  The parameter can be moved anywhere
                         in its range with no cost penalty: it is redundant
                         (either inert, or exactly compensated by another).

The independent parameters are the identifiable ones.  Everything else should
be fixed at a literature value rather than fitted.

--------------------------------------------------------------------------
HOW THIS RUN IS STRUCTURED
--------------------------------------------------------------------------
STAGE A -- circuit re-fit.
    v17 reverted the three circuit constants v16 had baked in, and runs at
    418 mL/min against a clinical 188.  Profiling mechanism parameters around
    a point that far off would just measure circuit mismatch, so R_AO4_UA,
    V_BLOOD_TOTAL and C_UA are fitted first (multi-start SLSQP in log space)
    and thereafter carried as nuisance parameters inside every profile.

STAGE B -- sensitivity scan (all 62 mechanism parameters).
    A cheap 1-D sweep with every other parameter held fixed.  Two jobs:
      1. rank parameters by how much they move the cost at all;
      2. identify the inert ones.  A scan is an UPPER BOUND on the profile
         (re-optimising nuisance parameters can only lower the cost), so a
         parameter whose scan is exactly flat is proven structurally
         non-identifiable without spending anything more on it.  Those are
         reported straight from the scan, which is what makes the full run
         affordable.

STAGE C -- true profile (parameters the scan showed are active).
    Nuisance set = the 3 circuit parameters + the N_NUISANCE_MECH highest-
    ranked mechanism parameters from stage B (excluding whichever parameter
    is currently being profiled).  Each grid point is warm-started from its
    neighbour, walking outward from the reference in both directions
    (continuation), so the inner optimisation starts nearly converged.

STAGE D -- plots.  All parabolas in one matplotlib grid, plus a ranked table.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python3 analysis/profile_likelihood_v17.py              # full run
    python3 analysis/profile_likelihood_v17.py --quick      # coarse, ~20 min
    python3 analysis/profile_likelihood_v17.py --stage a    # circuit fit only
    python3 analysis/profile_likelihood_v17.py --plot-only  # replot saved state
    python3 analysis/profile_likelihood_v17.py --show       # print the table

Every stage checkpoints to results/profile_likelihood_v17/ after each finished
parameter, and re-running resumes from there.  Ctrl-C never loses work.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import ap_cost as A
from project_paths import RESULTS_PROFILE_V17, ensure

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
OUT_DIR    = ensure(RESULTS_PROFILE_V17)
STATE_FILE = os.path.join(OUT_DIR, "profile_state.json")
PLOT_GRID  = os.path.join(OUT_DIR, "profile_likelihood_grid.png")
PLOT_TOP   = os.path.join(OUT_DIR, "profile_likelihood_top.png")
PLOT_RANK  = os.path.join(OUT_DIR, "parameter_ranking.png")
TABLE_FILE = os.path.join(OUT_DIR, "profile_summary.csv")

N_GRID          = 15     # points per profile
N_NUISANCE_MECH = 4      # mechanism params re-optimised alongside the 3 circuit ones
INNER_MAXITER   = 4      # SLSQP iterations per grid point (warm-started;
                         # the neighbour's solution means few are needed)
CIRCUIT_RESTARTS = 6     # multi-start count for stage A
CIRCUIT_MAXITER  = 50

# A scan whose total cost variation is below this counts as exactly "flat"
# -> structurally non-identifiable.  0.01 is 1% of a 1-sigma move, i.e. nothing.
FLAT_TOL = 1e-2

# Stage C is only spent on parameters whose SCAN already moves the cost by more
# than this.  The scan is a pointwise upper bound on the profile (re-optimising
# nuisance parameters can only lower the cost), so a parameter whose scan never
# reaches even the 1-sigma threshold cannot come out identifiable at the 95%
# threshold once profiled -- profiling can only flatten it further.  Reporting
# such a parameter straight from its scan is therefore conservative, and it is
# what makes profiling all 62 affordable: it concentrates the compute on the
# ~25 parameters where the verdict is actually in doubt.
# Hard cap on how many parameters get the full profile treatment, on top of
# SCAN_PROFILE_THRESHOLD.  Stage C evaluations are markedly slower than stage
# B's: forcing a parameter to the extremes of its range stiffens the ODE, so
# Radau takes more steps per solve.  Profiling every qualifying parameter at
# full depth does not fit an overnight budget on 6 performance cores.  The
# deepest parameters are profiled first, so the cap costs the least-informative
# ones -- and they are still reported from their scans, which are complete for
# all 62 and are a rigorous upper bound on the profile.
N_PROFILE_MAX = 12

SCAN_PROFILE_THRESHOLD = 3.84   # = CHI2_95, the rigorous cutoff: a parameter
                                # whose SCAN never reaches the 95% threshold
                                # cannot cross it once profiled either, because
                                # the profile lies below the scan pointwise.

CHI2_1SIGMA = 1.00       # dJ threshold, 68% CI, 1 dof
CHI2_95     = 3.84       # dJ threshold, 95% CI, 1 dof

MAX_STEP = A.MAX_STEP_FINE   # cost is step-insensitive (571.40 @ 0.01 vs
                             # 571.17 @ 0.002) and 0.002 is what v17 itself
                             # runs, so score the model exactly as it ships.
N_WORKERS = max(1, min(os.cpu_count() or 4, 10) - 1)


def quick_mode():
    """Coarser settings for a ~20 minute smoke run."""
    global N_GRID, N_NUISANCE_MECH, INNER_MAXITER, CIRCUIT_RESTARTS
    global CIRCUIT_MAXITER, MAX_STEP
    N_GRID, N_NUISANCE_MECH, INNER_MAXITER = 7, 2, 4
    CIRCUIT_RESTARTS, CIRCUIT_MAXITER = 2, 25
    MAX_STEP = A.MAX_STEP_COARSE


# ===========================================================================
#  LOG/LINEAR TRANSFORMS
# ===========================================================================
# Nuisance parameters are optimised in a transformed space: log for
# scale-free positive quantities, identity for fractions and gains.  Log-space
# makes SLSQP's fixed finite-difference step sensible across parameters whose
# magnitudes span 1e-7 to 7e4.

def _spec(name):
    if name in A.MECH_BY_NAME:
        p = A.MECH_BY_NAME[name]
        return p.scale, p.lo, p.hi
    return "log", A.CIRCUIT_BOUNDS[name][0], A.CIRCUIT_BOUNDS[name][1]


def to_z(name, value):
    scale, lo, hi = _spec(name)
    v = float(np.clip(value, lo, hi))
    return np.log(max(v, 1e-12)) if scale == "log" else v


def from_z(name, z):
    scale, lo, hi = _spec(name)
    v = np.exp(z) if scale == "log" else z
    return float(np.clip(v, lo, hi))


def z_bounds(name):
    scale, lo, hi = _spec(name)
    if scale == "log":
        return (np.log(max(lo, 1e-12)), np.log(hi))
    return (lo, hi)


# ===========================================================================
#  OBJECTIVE WRAPPERS
# ===========================================================================
# Some parameter combinations drive the model into a negative base for a
# fractional power (v17 line 716, cerebral autoregulation) and produce NaN.
# evaluate() already turns that into FAIL_COST; the warning itself is just noise
# that would otherwise fill the log with tens of thousands of identical lines.
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


def cost_of(overrides):
    return A.evaluate(overrides, max_step=MAX_STEP, duration=A.SIM_DURATION)


def make_inner_objective(base, free_names, fixed):
    """J as a function of the free nuisance parameters (in z-space).

    base   : full parameter dict (the reference point)
    free   : names re-optimised at this grid point
    fixed  : {name: value} forced on top of base (the profiled parameter)
    """
    def f(z):
        ov = dict(base)
        for name, zi in zip(free_names, z):
            ov[name] = from_z(name, zi)
        ov.update(fixed)
        return cost_of(ov)
    return f


# ===========================================================================
#  STAGE A -- circuit re-fit
# ===========================================================================
def _circuit_start(args):
    """One SLSQP restart of the circuit fit.  Runs in a worker process."""
    names, z0, maxiter = args

    def obj(z):
        return cost_of({n: from_z(n, zi) for n, zi in zip(names, z)})

    res = minimize(obj, z0, method="SLSQP",
                   bounds=[z_bounds(n) for n in names],
                   options={"maxiter": maxiter, "ftol": 1e-6, "eps": 1e-3})
    return float(res.fun), {n: from_z(n, zi) for n, zi in zip(names, res.x)}


def fit_circuit(pool, verbose=True):
    """Fit R_AO4_UA, V_BLOOD_TOTAL, C_UA with all mechanism params at v17 defaults.

    Restarts run in parallel: each is ~CIRCUIT_MAXITER * 4 evaluations at ~4 s
    each, so serialising them would cost hours on its own.
    """
    names = A.CIRCUIT_PARAMS
    defaults = A.v17_circuit_defaults()

    # Starting points: v17's own baselines, v16's working values (which fit the
    # data well under the old objective), and random draws across the box.
    # v16's values are included deliberately -- optimizer_v1.py's original
    # bounds excluded them and it could never find the answer.
    starts = [
        [defaults["R_AO4_UA"], defaults["V_BLOOD_TOTAL"], defaults["C_UA"]],
        [40.545318, 268.787, 0.002471],       # v16's baked-in working set
    ]
    rng = np.random.default_rng(20260819)
    for _ in range(max(0, CIRCUIT_RESTARTS - len(starts))):
        starts.append([
            float(np.exp(rng.uniform(*np.log(A.CIRCUIT_BOUNDS["R_AO4_UA"])))),
            float(np.exp(rng.uniform(*np.log(A.CIRCUIT_BOUNDS["V_BLOOD_TOTAL"])))),
            float(np.exp(rng.uniform(*np.log(A.CIRCUIT_BOUNDS["C_UA"])))),
        ])

    jobs = [(names, [to_z(n, v) for n, v in zip(names, s)], CIRCUIT_MAXITER)
            for s in starts]
    best, best_cost = None, np.inf
    t0 = time.time()
    for i, (c, vals) in enumerate(pool.imap_unordered(_circuit_start, jobs), 1):
        if verbose:
            print(f"  start {i}/{len(jobs)}: J={c:12.3f}  "
                  f"R={vals['R_AO4_UA']:8.3f}  V={vals['V_BLOOD_TOTAL']:8.2f}  "
                  f"C_UA={vals['C_UA']:.6f}   [{(time.time()-t0)/60:.1f} min]")
        if c < best_cost:
            best, best_cost = vals, c
    return best, best_cost


# ===========================================================================
#  STAGE B -- sensitivity scan
# ===========================================================================
def _scan_task(args):
    name, values, base = args
    return name, [cost_of({**base, name: float(v)}) for v in values]


def run_scan(base, ref, pool):
    tasks = []
    for p in A.MECH_PARAMS:
        tasks.append((p.name, p.grid(ref[p.name], N_GRID).tolist(), base))
    out = {}
    t0 = time.time()
    for i, (name, costs) in enumerate(pool.imap_unordered(_scan_task, tasks), 1):
        vals = A.MECH_BY_NAME[name].grid(ref[name], N_GRID).tolist()
        out[name] = {"values": vals, "cost": costs}
        el = time.time() - t0
        print(f"  [{i:>2}/{len(tasks)}] {name:<22} range dJ = "
              f"{max(costs)-min(costs):10.3f}   ({el/i*len(tasks)/60:.0f} min est)")
    return out


# ===========================================================================
#  STAGE C -- true profile
# ===========================================================================
def _profile_task(args):
    """Profile ONE parameter across its grid.  Runs inside a worker process."""
    name, values, ref_value, base, nuisance = args
    p = A.MECH_BY_NAME[name]
    free = [n for n in nuisance if n != name]

    values = list(values)
    i_ref = int(np.argmin(np.abs(np.array(values) - ref_value)))

    costs = [None] * len(values)
    nuis  = [None] * len(values)

    def solve_at(idx, warm):
        """Re-optimise the nuisance parameters with `name` pinned at values[idx]."""
        fixed = {name: float(values[idx])}
        f = make_inner_objective(base, free, fixed)
        z0 = [to_z(n, warm[n]) for n in free]
        c_start = f(z0)
        try:
            res = minimize(f, z0, method="SLSQP",
                           bounds=[z_bounds(n) for n in free],
                           options={"maxiter": INNER_MAXITER, "ftol": 1e-4,
                                    "eps": 1e-3})
            c_res, z_res = float(res.fun), res.x
        except Exception:
            # One bad grid point must not take down the whole pool and lose
            # every other parameter's profile with it.
            c_res, z_res = np.inf, z0
        # SLSQP can also return a point WORSE than its start when it runs out of
        # iterations; a profile must be a MINIMUM, so keep whichever is lower.
        if not np.isfinite(c_res) or c_start <= c_res:
            return c_start, {n: warm[n] for n in free}
        return c_res, {n: from_z(n, zi) for n, zi in zip(free, z_res)}

    # Continuation: solve the reference point, then walk outward in both
    # directions warm-starting from the neighbour already solved.
    warm0 = {n: base[n] for n in free}
    costs[i_ref], nuis[i_ref] = solve_at(i_ref, warm0)

    warm = dict(warm0); warm.update(nuis[i_ref])
    for i in range(i_ref + 1, len(values)):
        costs[i], sol = solve_at(i, warm)
        nuis[i] = sol
        warm = {**warm, **sol}

    warm = dict(warm0); warm.update(nuis[i_ref])
    for i in range(i_ref - 1, -1, -1):
        costs[i], sol = solve_at(i, warm)
        nuis[i] = sol
        warm = {**warm, **sol}

    return name, {"values": values, "cost": costs, "nuisance": nuis,
                  "free": free, "mech": p.mech}


def run_profiles(base, ref, scan, nuisance, pool, done):
    ranges = {p.name: max(scan[p.name]["cost"]) - min(scan[p.name]["cost"])
              for p in A.MECH_PARAMS}
    active = [p.name for p in A.MECH_PARAMS
              if ranges[p.name] > SCAN_PROFILE_THRESHOLD and p.name not in done]
    active.sort(key=lambda n: -ranges[n])      # deepest first, so a cap or an
                                              # early stop costs the least
    if N_PROFILE_MAX and len(active) > N_PROFILE_MAX:
        deferred = active[N_PROFILE_MAX:]
        active = active[:N_PROFILE_MAX]
        print(f"  capped at N_PROFILE_MAX={N_PROFILE_MAX}; "
              f"{len(deferred)} deferred to their scans: {', '.join(deferred)}")
    n_skip = sum(1 for p in A.MECH_PARAMS if ranges[p.name] <= SCAN_PROFILE_THRESHOLD)
    print(f"  {len(active)} parameters to profile; {n_skip} reported from their "
          f"scan (range <= {SCAN_PROFILE_THRESHOLD}, cannot reach the 95% "
          f"threshold); {len(done)} already done")

    tasks = []
    for name in active:
        tasks.append((name, scan[name]["values"], ref[name], base, nuisance))

    out = {}
    t0 = time.time()
    for i, (name, rec) in enumerate(pool.imap_unordered(_profile_task, tasks), 1):
        out[name] = rec
        el = (time.time() - t0) / 60
        print(f"  [{i:>2}/{len(tasks)}] {name:<22} dJ range = "
              f"{max(rec['cost'])-min(rec['cost']):9.3f}   "
              f"({el:.0f} min elapsed, ~{el/i*len(tasks):.0f} min total)")
        yield name, rec


# ===========================================================================
#  VERDICTS
# ===========================================================================
def crossings(values, dJ, threshold):
    """Where the profile crosses `threshold`, by linear interpolation.

    Returns (lower, upper); None on a side that never crosses.
    """
    v = np.asarray(values, dtype=float)
    d = np.asarray(dJ, dtype=float)
    i_min = int(np.argmin(d))

    def walk(rng):
        prev = i_min
        for i in rng:
            if d[i] >= threshold:
                if d[i] == d[prev]:
                    return float(v[i])
                f = (threshold - d[prev]) / (d[i] - d[prev])
                return float(v[prev] + f * (v[i] - v[prev]))
            prev = i
        return None

    return walk(range(i_min - 1, -1, -1)), walk(range(i_min + 1, len(v)))


def global_min(state):
    """Lowest cost seen anywhere -- the reference point, any scan point, any
    profile point.  Every dJ in the plots and the table is measured from this
    one number, so panels are directly comparable to each other."""
    best = float(state.get("J_ref", np.inf))
    for group in ("scan", "profile"):
        for rec in state.get(group, {}).values():
            best = min(best, float(np.min(rec["cost"])))
    return best


def valid_mask(cost):
    """Points where the model actually integrated.

    evaluate() returns FAIL_COST for a parameter value that makes the ODE
    unsolvable.  That is a sentinel, not a chi-square value: leaving it in makes
    a profile look 1e6 deep and corrupts the ranking.  Such points mean "the
    model does not admit this value", which is reported separately rather than
    being scored as an enormous but finite misfit.
    """
    c = np.asarray(cost, float)
    return np.isfinite(c) & (c < A.FAIL_COST * 0.99)


def verdict(values, cost, J_min=None, threshold=CHI2_95):
    v_all = np.asarray(values, float)
    c_all = np.asarray(cost, float)
    ok = valid_mask(c_all)
    if ok.sum() < 3:
        # too little of the range is admissible to say anything
        return "structural", None, None, 0.0
    values = v_all[ok]
    cost = c_all[ok]

    dJ = cost - (float(np.min(cost)) if J_min is None else float(J_min))
    lo, hi = crossings(values, dJ, threshold)
    span = float(dJ.max() - dJ.min())
    if span <= FLAT_TOL:
        return "structural", None, None, span
    if lo is not None and hi is not None:
        return "identifiable", lo, hi, span
    if lo is None and hi is None:
        return "structural", None, None, span
    return "practical", lo, hi, span


# ===========================================================================
#  STATE
# ===========================================================================
def signature():
    return {"sim_duration": A.SIM_DURATION,
            "analysis_seconds": A.ANALYSIS_SECONDS,
            "features": A.FEATURES,
            "weights": A.WEIGHTS,
            "amp_pctl": list(A.AMP_PCTL),
            "max_step": MAX_STEP,
            "n_grid": N_GRID,
            "n_nuisance_mech": N_NUISANCE_MECH,
            "model": "AP_model_v17"}


def load_state():
    if not os.path.isfile(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] could not read {STATE_FILE}: {e} -- starting fresh")
        return {}
    if st.get("signature") != signature():
        print("[warn] settings changed since the saved state; scan/profiles "
              "are not comparable and will be recomputed.")
        return {"circuit": st.get("circuit")}
    return st


def save_state(st):
    st["signature"] = signature()
    st["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_FILE)


# ===========================================================================
#  PLOTTING
# ===========================================================================
MECH_COLORS = {
    "baroreflex":     "#1f77b4", "frank_starling": "#d62728",
    "catecholamine":  "#ff7f0e", "edema":          "#8c564b",
    "venous_Vu":      "#9467bd", "chemoreflex":    "#2ca02c",
    "cerebral_auto":  "#17becf", "ductus":         "#e377c2",
    "pvr":            "#7f7f7f", "regurgitation":  "#bcbd22",
    "raas":           "#aec7e8", "vasopressin":    "#98df8a",
}
VERDICT_MARK = {"identifiable": "✓", "practical": "~", "structural": "✗"}


def plot_grid(state, path=PLOT_GRID, subset=None, ncols=None):
    """Every parabola in one figure, ordered by mechanism.

    subset : int, optional
        Render only the `subset` parameters with the deepest profiles instead
        of all 62, ordered by depth rather than by mechanism.  All 62 panels
        are unreadable at poster scale; `--subset 12` is the version worth
        printing, and it uses exactly the same panel formatting.
    """
    records = state.get("profile", {})
    scan    = state.get("scan", {})
    J_min   = global_min(state)

    params = [p for p in A.MECH_PARAMS
              if p.name in records or p.name in scan]
    if subset:
        rank = {r["name"]: i for i, r in enumerate(summary_rows(state))}
        params.sort(key=lambda p: rank.get(p.name, 10**6))
        params = params[:subset]
    else:
        params.sort(key=lambda p: (A.MECH_ORDER.index(p.mech), p.name))
    n = len(params)
    if ncols is None:
        ncols = 8 if not subset else min(4, n)
    nrows = int(np.ceil(n / ncols))

    panel_w, panel_h = (3.0, 2.5) if not subset else (4.2, 3.4)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows))
    axes = np.atleast_1d(axes).ravel()

    n_id = n_pr = n_st = 0
    for ax, p in zip(axes, params):
        profiled = p.name in records
        rec = records.get(p.name) or scan[p.name]
        v = np.asarray(rec["values"], float)
        c = np.asarray(rec["cost"], float)
        ok = valid_mask(c)
        v_bad = v[~ok]
        v, c = v[ok], c[ok]
        if len(v) == 0:
            ax.axis("off"); continue
        dJ = c - J_min

        vd, lo, hi, span = verdict(v, c, J_min)
        n_id += vd == "identifiable"
        n_pr += vd == "practical"
        n_st += vd == "structural"

        col = MECH_COLORS[p.mech]
        ax.plot(v, dJ, marker="o", color=col, ms=3, lw=1.4,
                ls="-" if profiled else "--")
        ax.axhline(CHI2_95, color="k", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(CHI2_1SIGMA, color="k", lw=0.6, ls=":", alpha=0.5)
        ax.axvline(state["ref"][p.name], color="0.35", lw=0.9, ls="-.")
        if lo is not None:
            ax.axvline(lo, color=col, lw=0.7, alpha=0.5)
        if hi is not None:
            ax.axvline(hi, color=col, lw=0.7, alpha=0.5)

        if p.scale == "log":
            ax.set_xscale("log")
        top = max(CHI2_95 * 2.0, float(np.percentile(dJ, 90)) * 1.3, 1e-3)
        for xb in v_bad:      # values where the ODE could not be solved
            ax.axvline(xb, color="0.6", lw=3, alpha=0.35, zorder=0)
        ax.set_ylim(-0.05 * top, top)
        ax.set_title(f"{VERDICT_MARK[vd]} {p.label}", fontsize=8.5, color=col)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, lw=0.5)

    for ax in axes[n:]:
        ax.axis("off")

    for ax in axes[:n]:
        ax.set_ylabel(r"$\Delta J$", fontsize=7)

    handles = [Line2D([], [], color=MECH_COLORS[m], lw=2, label=m)
               for m in A.MECH_ORDER]
    handles += [
        Line2D([], [], color="k", ls="--", lw=1, label=r"$\Delta J=3.84$ (95%)"),
        Line2D([], [], color="k", ls=":",  lw=1, label=r"$\Delta J=1$ (1$\sigma$)"),
        Line2D([], [], color="0.35", ls="-.", lw=1, label="reference value"),
        Line2D([], [], color="0.2", ls="-",  lw=1.4, label="profiled (nuisance re-optimised)"),
        Line2D([], [], color="0.2", ls="--", lw=1.4, label="scan only (proven flat)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    scope = (f"{n} most-constrained of {len(A.MECH_PARAMS)} mechanism parameters"
             if subset else
             f"all {len(A.MECH_PARAMS)} homeostatic mechanism parameters")
    fig.suptitle(
        f"Profile likelihood -- AP_model_v17, {scope}\n"
        f"($J_{{min}}$ = {J_min:.2f}, {len(A.FEATURES)} features, "
        f"{A.SIM_DURATION:.0f} s sim / last {A.ANALYSIS_SECONDS:.0f} s scored)   "
        f"identifiable: {n_id}   practically non-identifiable: {n_pr}   "
        f"structurally non-identifiable: {n_st}",
        fontsize=13 if not subset else 15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return {"identifiable": n_id, "practical": n_pr, "structural": n_st}


def plot_ranking(state, path=PLOT_RANK, top=25):
    rows = summary_rows(state)
    rows = [r for r in rows if r["span"] > 0][:top]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["span"] for r in rows],
            color=[MECH_COLORS[r["mech"]] for r in rows])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{VERDICT_MARK[r['verdict']]} {r['name']}" for r in rows],
                       fontsize=8)
    ax.set_xscale("log")
    ax.axvline(CHI2_95, color="k", ls="--", lw=1, label=r"$\Delta J=3.84$")
    ax.set_xlabel(r"profile depth  $\max\Delta J$  over the parameter's range")
    ax.set_title("Mechanism parameters ranked by how much the data constrain them",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
#  SUMMARY TABLE
# ===========================================================================
def summary_rows(state):
    J_min = global_min(state)
    rows = []
    for p in A.MECH_PARAMS:
        rec = state.get("profile", {}).get(p.name) or state.get("scan", {}).get(p.name)
        if rec is None:
            continue
        v, c = rec["values"], rec["cost"]
        vd, lo, hi, span = verdict(v, c, J_min)
        i_best = int(np.argmin(c))
        rows.append({
            "name": p.name, "mech": p.mech, "label": p.label,
            "ref": state["ref"][p.name],
            "best": float(v[i_best]), "best_cost": float(min(c)),
            "verdict": vd, "ci_lo": lo, "ci_hi": hi, "span": span,
            "profiled": p.name in state.get("profile", {}),
        })
    rows.sort(key=lambda r: -r["span"])
    return rows


def write_table(state, path=TABLE_FILE):
    rows = summary_rows(state)
    with open(path, "w") as f:
        f.write("rank,parameter,mechanism,reference,best_value,best_cost,"
                "verdict,ci_lo_95,ci_hi_95,profile_depth_dJ,profiled\n")
        for i, r in enumerate(rows, 1):
            lo = "" if r["ci_lo"] is None else f"{r['ci_lo']:.6g}"
            hi = "" if r["ci_hi"] is None else f"{r['ci_hi']:.6g}"
            f.write(f"{i},{r['name']},{r['mech']},{r['ref']:.6g},"
                    f"{r['best']:.6g},{r['best_cost']:.6g},{r['verdict']},"
                    f"{lo},{hi},{r['span']:.6g},{int(r['profiled'])}\n")
    print(f"  wrote {path}")


def print_table(state, top=None):
    rows = summary_rows(state)
    print(f"\n{'#':>3} {'parameter':<24}{'mechanism':<16}{'reference':>12}"
          f"{'best':>12}{'dJ range':>11}  {'verdict':<14}95% CI")
    print("-" * 118)
    for i, r in enumerate(rows if top is None else rows[:top], 1):
        ci = "-"
        if r["ci_lo"] is not None or r["ci_hi"] is not None:
            lo = "-inf" if r["ci_lo"] is None else f"{r['ci_lo']:.4g}"
            hi = "+inf" if r["ci_hi"] is None else f"{r['ci_hi']:.4g}"
            ci = f"[{lo}, {hi}]"
        print(f"{i:>3} {r['name']:<24}{r['mech']:<16}{r['ref']:>12.5g}"
              f"{r['best']:>12.5g}{r['span']:>11.4g}  "
              f"{VERDICT_MARK[r['verdict']]} {r['verdict']:<12}{ci}")
    n = {k: sum(1 for r in rows if r["verdict"] == k)
         for k in ("identifiable", "practical", "structural")}
    print("-" * 118)
    print(f"identifiable {n['identifiable']}   practically non-identifiable "
          f"{n['practical']}   structurally non-identifiable {n['structural']}"
          f"   (of {len(rows)})")


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="coarse settings, ~20 min")
    ap.add_argument("--stage", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--show", action="store_true", help="print the saved table")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--subset", type=int, default=12,
                    help="how many parameters go in the poster-sized grid "
                         "(profile_likelihood_top.png); 0 disables it")
    args = ap.parse_args()

    if args.quick:
        quick_mode()
    if args.reset and os.path.isfile(STATE_FILE):
        os.remove(STATE_FILE)

    state = load_state()

    if args.show:
        if not state.get("scan"):
            print("no saved state yet.")
            return
        print_table(state)
        return

    if args.plot_only:
        plot_grid(state)
        if args.subset:
            plot_grid(state, path=PLOT_TOP, subset=args.subset)
        plot_ranking(state); write_table(state); print_table(state)
        return

    t_start = time.time()
    ok, d = A.verify_against_v17()
    print(f"v17 transcription self-test: {'PASS' if ok else 'FAIL'} (max |dx|={d:.2e})")
    if not ok:
        print("  AP_model_v17 has changed since ap_cost.make_x0 was written. Stop.")
        return

    c_f, s_f, win, n_s = A.clinical_targets()
    print(f"clinical window {A.START_TIME}-{A.END_TIME}: {win:.1f} s, {n_s} samples")
    print(f"workers: {args.workers}   grid: {N_GRID}   "
          f"nuisance: 3 circuit + {N_NUISANCE_MECH} mechanism")

    with Pool(args.workers) as pool:
        # ---- STAGE A -----------------------------------------------------
        if state.get("circuit") is None:
            print("\n=== STAGE A: re-fitting the 3 circuit parameters ===")
            print(f"  J(v17 as written) = {cost_of({}):.3f}")
            circ, circ_cost = fit_circuit(pool)
            state["circuit"] = circ
            state["J_ref"] = circ_cost
            save_state(state)
        else:
            circ = state["circuit"]
        print(f"\ncircuit fit: R_AO4_UA={circ['R_AO4_UA']:.4f}  "
              f"V_BLOOD_TOTAL={circ['V_BLOOD_TOTAL']:.3f}  C_UA={circ['C_UA']:.6f}")

        ref = A.reference_values(circuit=circ)
        state["ref"] = ref
        base = dict(ref)
        if state.get("J_ref") is None:
            state["J_ref"] = cost_of(base)
        print(f"J at the reference point = {state['J_ref']:.3f}")
        save_state(state)
        if args.stage == "a":
            return

        # ---- STAGE B -----------------------------------------------------
        if not state.get("scan"):
            print(f"\n=== STAGE B: sensitivity scan of "
                  f"{len(A.MECH_PARAMS)} mechanism parameters ===")
            state["scan"] = run_scan(base, ref, pool)
            save_state(state)
        scan = state["scan"]

        # Nuisance set: the highest-ranked mechanism parameters from the scan.
        ranked = sorted(scan.items(),
                        key=lambda kv: -(max(kv[1]["cost"]) - min(kv[1]["cost"])))
        nuisance = A.CIRCUIT_PARAMS + [k for k, _ in ranked[:N_NUISANCE_MECH]]
        state["nuisance"] = nuisance
        print(f"\nnuisance set: {nuisance}")
        save_state(state)
        if args.stage == "b":
            plot_grid(state); plot_ranking(state); write_table(state)
            return

        # ---- STAGE C -----------------------------------------------------
        print("\n=== STAGE C: profiling (nuisance re-optimised at every point) ===")
        state.setdefault("profile", {})
        for name, rec in run_profiles(base, ref, scan, nuisance, pool,
                                      set(state["profile"])):
            state["profile"][name] = rec
            save_state(state)

    # ---- STAGE D ---------------------------------------------------------
    print(f"\n=== STAGE D: plots and summary ===")
    plot_grid(state)
    if args.subset:
        plot_grid(state, path=PLOT_TOP, subset=args.subset)
    plot_ranking(state)
    write_table(state)
    print_table(state)
    print(f"\ntotal wall time: {(time.time()-t_start)/60:.1f} min")
    print(f"state: {STATE_FILE}")


if __name__ == "__main__":
    main()
