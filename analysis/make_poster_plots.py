"""
Poster figures: large type, no titles, written to poster/.

    python3 analysis/make_poster_plots.py

Three figures:
  1. poster/fig1_da_G_O2_profile.png   one parameter's likelihood curve
  2. poster/fig2_sensitivity_subset.png  sensitivity heatmap, trimmed
  3. poster/fig3_v15_vs_clinical.png     2x2, v15 against the recording
  4. poster/fig3_v18_vs_clinical.png     2x2, v18 against the recording

Figures 3 and 4 use identical panels, order and y-limits so the two model
versions can be read side by side on the poster.

None carry a title -- panel identity is carried by the axis labels, so each
figure can sit under its own poster heading.  Type is sized for viewing at
arm's length on a printed poster, not for a screen.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import ap_cost as A
from analysis import profile_likelihood_v17 as PL
from project_paths import ROOT, ensure

OUT = ensure(os.path.join(ROOT, "poster"))

# Poster type scale.  Everything is set once here so the three figures match.
plt.rcParams.update({
    "font.size":        22,
    "axes.labelsize":   26,
    "axes.titlesize":   26,
    "xtick.labelsize":  22,
    "ytick.labelsize":  22,
    "legend.fontsize":  20,
    "axes.linewidth":   2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "lines.linewidth":  3.0,
})


# ===========================================================================
#  FIGURE 1 -- one parameter's likelihood curve
# ===========================================================================
def fig1(param="da_G_O2", path=None, source="auto"):
    """One parameter's likelihood curve.

    source : "scan"    -- every other parameter held fixed (a likelihood scan)
             "profile" -- nuisance parameters re-optimised at each grid point
                          (a true profile likelihood)
             "auto"    -- use the profile if one exists, else the scan

    The distinction matters: a scan is a pointwise UPPER BOUND on the profile,
    so scans look far more parabolic -- and far more constrained -- than the
    truth.  For da_G_O2 the scan spans dJ ~ 19 while the profile spans 0.012.
    """
    default_name = {"scan": "fig1_da_G_O2_scan.png",
                    "profile": "fig1_da_G_O2_profile.png"}
    st = PL.load_state()
    J_min = PL.global_min(st)

    if source == "auto":
        source = "profile" if param in st.get("profile", {}) else "scan"
    if source == "profile" and param not in st.get("profile", {}):
        raise KeyError(f"{param} has not been profiled")
    profiled = source == "profile"
    path = path or os.path.join(OUT, default_name[source])
    rec = st["profile"][param] if profiled else st["scan"][param]
    p = A.MECH_BY_NAME[param]

    v = np.asarray(rec["values"], float)
    c = np.asarray(rec["cost"], float)
    ok = PL.valid_mask(c)
    v, c = v[ok], c[ok]
    dJ = c - J_min

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.plot(v, dJ, marker="o", ms=11, color="#c2185b", lw=3.5, zorder=3)

    ax.axhline(PL.CHI2_95, color="k", ls="--", lw=2.5, zorder=2)
    ax.axhline(PL.CHI2_1SIGMA, color="0.45", ls=":", lw=2.5, zorder=2)
    ax.axvline(st["ref"][param], color="0.35", ls="-.", lw=2.5, zorder=2)

    lo, hi = PL.crossings(v, dJ, PL.CHI2_95)
    for x in (lo, hi):
        if x is not None:
            ax.axvline(x, color="#c2185b", lw=2, alpha=0.55, zorder=1)
    if lo is not None and hi is not None:
        ax.axvspan(lo, hi, color="#c2185b", alpha=0.07, zorder=0)

    # Label the thresholds in the plot rather than in a title.
    xr = ax.get_xlim()
    ax.text(xr[1], PL.CHI2_95, " 95% CI ", ha="right", va="bottom", fontsize=20,
            color="k", fontweight="bold")
    ax.text(xr[1], PL.CHI2_1SIGMA, r" 1$\sigma$ ", ha="right", va="bottom",
            fontsize=20, color="0.35")

    ax.set_xlabel(r"Ductus O$_2$ constriction gain, $G_{O_2}$", labelpad=10)
    ax.set_ylabel("Worsening of fit to experimental data,\n"
                  r"$\Delta J$  ($\chi^2$ units)", labelpad=10)
    ax.grid(alpha=0.3, lw=1.2)
    ax.set_ylim(bottom=-0.4)
    if p.scale == "log":
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    kind = "profile likelihood" if profiled else "likelihood scan"
    print(f"  wrote {path}   [{kind}]  95% CI = "
          f"[{'-inf' if lo is None else f'{lo:.3f}'}, "
          f"{'+inf' if hi is None else f'{hi:.3f}'}]")
    return profiled


# ===========================================================================
#  FIGURE 2 -- sensitivity heatmap, trimmed
# ===========================================================================
DROP_ROWS = ["p_pre_ecmo", "p_ua_mean"]
KEEP_LAST_N_COLS = 5

PRETTY_FEATURE = {
    "q_mean":       "Mean AP flow",
    "q_amp":        "AP flow amplitude",
    "dp_pump_mean": "Pump $\\Delta P$",
    "dp_ecmo":      "ECMO $\\Delta P$",
    "dp_restr":     "Restrictor $\\Delta P$",
    "p_pre_ecmo":   "Pre-ECMO pressure",
    "p_ua_mean":    "UA pressure",
}
PRETTY_PARAM = {
    "fs_V_ref_frac": r"$V_{ref}$ frac",
    "fs_alpha":      r"$\alpha_{FS}$",
    "E_max_L":       r"$E_{max,LV}$",
    "E_max_R":       r"$E_{max,RV}$",
    "cat_C_peak":    r"$C_{peak}$",
    "HR_intrinsic":  r"$HR_{intr}$",
    "R_AO4_UA":      r"$R_{AO4-UA}$",
    "V_BLOOD_TOTAL": r"$V_{blood}$",
    "C_UA":          r"$C_{UA}$",
}


def fig2(path=None):
    import json
    path = path or os.path.join(OUT, "fig2_sensitivity_subset.png")
    with open(os.path.join(ROOT, "results", "sensitivity_v17",
                           "fitted_parameters.json")) as f:
        fit = json.load(f)

    S = np.array(fit["sensitivity_matrix"])
    names = fit["sensitivity_names"]
    feats = fit["features"]

    keep_r = [i for i, f in enumerate(feats) if f not in DROP_ROWS]
    keep_c = list(range(len(names)))[-KEEP_LAST_N_COLS:]

    S = S[np.ix_(keep_r, keep_c)]
    row_lab = [PRETTY_FEATURE.get(feats[i], feats[i]) for i in keep_r]
    col_lab = [PRETTY_PARAM.get(names[j], names[j]) for j in keep_c]

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    vmax = float(np.nanmax(np.abs(S))) or 1.0
    im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_lab)))
    ax.set_xticklabels(col_lab, rotation=35, ha="right")
    ax.set_yticks(range(len(row_lab)))
    ax.set_yticklabels(row_lab)

    for j in range(S.shape[0]):
        for i in range(S.shape[1]):
            val = S[j, i]
            ax.text(i, j, f"{val:.2f}", ha="center", va="center", fontsize=20,
                    color="w" if abs(val) > 0.55 * vmax else "k",
                    fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\sigma$ of feature per 1% parameter change",
                 fontsize=22, labelpad=14)
    cb.ax.tick_params(labelsize=20)

    ax.set_xticks(np.arange(-.5, S.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, S.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}   ({S.shape[0]} rows x {S.shape[1]} cols)")
    print(f"     rows: {row_lab}")
    print(f"     cols: {col_lab}")


# ===========================================================================
#  FIGURE 3 -- v15 and v18 against the recording
# ===========================================================================
def _run_models(duration):
    """Simulate v15 and v18 once and return the traces both figures need."""
    import contextlib, io
    import models.AP_model_v15_temp as M15
    import models.AP_model_v18_temp as M18

    with contextlib.redirect_stdout(io.StringIO()):
        m15 = M15.ArtificialPlacentaModel(reflex_on=True)
        s15 = m15.run_simulation(duration=duration)
        _, q15, pre15, post15, restr15, ua15 = M15.post_process(s15, m15)

        m18 = M18.ArtificialPlacentaModel()
        s18 = m18.run_simulation(duration=duration)
        _, q18, pre18, post18, restr18, ua18 = M18.post_process(s18, m18)

    return {
        "v15": dict(t=s15.t, ua=ua15, pre=pre15, restr=restr15, q=q15 * 60.0),
        "v18": dict(t=s18.t, ua=ua18, pre=pre18, restr=restr18, q=q18 * 60.0),
    }


# Panel layout shared by both figures, so v15 and v18 stay directly comparable:
# same quantities, same order, same y-limits.
#   key        clinical signal attr          y label                            ylim
# The unit goes on a second line: a rotated y-label written on one line is
# nearly as tall as the axes at poster type size and gets clipped at the
# figure edge.  Two lines halve its vertical extent.
PANELS = [
    ("ua",    "_p_pre_pump", "Pre-pump pressure\n(mmHg)",        (-60, 0)),
    ("pre",   "_p_pre_ecmo", "Pre-ECMO pressure\n(mmHg)",        (10, 60)),
    ("restr", "_p_post_res", "Post-restrictor pressure\n(mmHg)", (10, 50)),
    ("q",     None,          "AP flow\n(mL/min)",                (150, 250)),
]

# Both versions use the same clinical-green / model-red pairing, so the two
# figures are read the same way and only the model curve's shape differs.
C_CLIN = "#1a9850"
C_MODEL = {"v15": "#d62728", "v18": "#d62728"}


def fig3(version, traces=None, clin=None, mask=None, t_c=None,
         duration=None, path=None):
    """One 2x2 figure comparing a single model version with the recording.

    The clinical pre-pump transducer sits at the umbilical-artery node, so its
    model counterpart is P_UA -- the same mapping AP_model_v17/v18's own
    comparison_plots uses for its 'Pre-Pump P' panel.
    """
    path = path or os.path.join(OUT, f"fig3_{version}_vs_clinical.png")

    if clin is None:
        clin = A.ClinicalData(A.clinical_csv())
        mask = clin.get_time_window_indices(A.START_TIME, A.END_TIME)
        t_c = clin._times[mask] - clin._times[mask][0]
    duration = duration or float(t_c[-1])
    if traces is None:
        traces = _run_models(duration)
    tr = traces[version]

    fig, axes = plt.subplots(2, 2, figsize=(19.5, 13.5))
    handles = labels = None
    for ax, (key, clin_attr, ylab, ylim) in zip(axes.ravel(), PANELS):
        y_c = (clin._flow_ml_s[mask] * 60.0 if clin_attr is None
               else getattr(clin, clin_attr)[mask])
        ax.plot(t_c, y_c, color=C_CLIN, lw=3.0, label="Clinical", zorder=3)
        # The model curve is labelled by what it is, not by which version, so
        # the v15 and v18 figures carry an identical legend.
        ax.plot(tr["t"], tr[key], color=C_MODEL[version], lw=2.6,
                label="LPM model", zorder=4)
        ax.set_xlabel("Time (s)", labelpad=8)
        ax.set_ylabel(ylab, labelpad=12, linespacing=1.6)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, duration)
        ax.grid(alpha=0.3, lw=1.2)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # One horizontal legend along the bottom.  No title -- the poster supplies
    # that.  Labelling the model curve "LPM model" rather than by version keeps
    # the v15 and v18 figures visually interchangeable.
    # rect reserves a band at the bottom for the legend and a sliver at the top
    # so the rotated y-labels are not clipped by the figure edge; the legend is
    # then drawn INSIDE the reserved band rather than below it, which would
    # leave a gap.
    fig.tight_layout(pad=2.6, rect=[0, 0.045, 1, 0.985])
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=28, handlelength=2.0, columnspacing=4.0,
               borderaxespad=0.0, bbox_to_anchor=(0.5, 0.004))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}   ({version}, {duration:.1f} s)")


def fig3_both():
    """Both 2x2 figures from a single pair of simulations."""
    clin = A.ClinicalData(A.clinical_csv())
    mask = clin.get_time_window_indices(A.START_TIME, A.END_TIME)
    t_c = clin._times[mask] - clin._times[mask][0]
    duration = float(t_c[-1])
    traces = _run_models(duration)
    for version in ("v15", "v18"):
        fig3(version, traces=traces, clin=clin, mask=mask, t_c=t_c,
             duration=duration)


if __name__ == "__main__":
    print("Poster figures ->", OUT)
    fig1(source="scan")
    fig1(source="profile")
    fig2()
    fig3_both()
