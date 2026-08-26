"""
Shared objective, simulator, and parameter registry for AP_model_v17 analyses.

Imported by:
    analysis/profile_likelihood_v17.py    (identifiability of mechanism params)
    analysis/sensitivity_v17.py           (re-fit of the identifiable ones)

--------------------------------------------------------------------------
THE COST FUNCTION
--------------------------------------------------------------------------
A chi-square-style normalised FEATURE cost, not a pointwise time-series SSR.

    J(theta) = sum_f  w_f * ( (m_f(theta) - c_f) / sigma_f )^2

where m_f is a scalar feature of the simulated trace, c_f the same feature of
the clinical recording, and sigma_f the clinical within-window standard
deviation of that feature (its natural noise scale).

Why features and not pointwise residuals: the model is not phase-locked to the
recording -- there is no gating signal, so the simulated cardiac cycle drifts
in phase against the clinical one.  A pointwise SSR would then be dominated by
phase error rather than by physiology, and every mechanism profile would sit on
top of that noise floor.  The features below are all phase-invariant.

Why squared and sigma-normalised (optimizer_v1.py used an un-squared relative
error): profile likelihood needs the objective to behave as a chi-square, so
that the standard thresholds apply directly --

    dJ = 1.00  ->  1-sigma (68%) confidence interval, 1 dof
    dJ = 3.84  ->  95% confidence interval, 1 dof

Those thresholds are what turn a parabola into a verdict:
    both sides cross      -> identifiable        (finite CI)
    one side stays flat   -> practically non-identifiable
    neither side crosses  -> structurally non-identifiable (parameter is inert)

--------------------------------------------------------------------------
THE FEATURES  (8 terms)
--------------------------------------------------------------------------
All computed over the last ANALYSIS_SECONDS of a SIM_DURATION-second run,
resampled onto a uniform RESAMPLE_HZ grid first so every statistic is a true
time average.  (Radau is adaptive: it clusters steps around the systolic
upstroke, so np.mean on the raw solver grid is sample-weighted, not
time-weighted, and answers a different question than the 10 Hz clinical trace.
This is the same correction already applied in optimizer_v1.py.)

    q_mean          mean AP flow                        [mL/min]
    q_amp           AP flow pulse amplitude             [mL/min]
    dp_pump_mean    mean pump pressure rise             [mmHg]
    dp_pump_amp     pump pressure rise amplitude        [mmHg]
    dp_ecmo         mean pre-ECMO -> post-ECMO drop     [mmHg]
    dp_restr        mean post-ECMO -> post-restrictor   [mmHg]
    p_pre_ecmo      mean pre-ECMO pressure              [mmHg]
    p_ua_mean       mean UA (pre-pump) pressure         [mmHg]

Amplitudes are percentile-based (AMP_PCTL) so one noisy clinical sample cannot
define them -- again matching optimizer_v1.py.
"""

import contextlib
import io
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models.AP_model_v17 as M17
from models.AP_model_v17 import (ArtificialPlacentaModel, ClinicalData,
                                 MechanismToggles, START_TIME, END_TIME,
                                 N_STATES, I_AR, I_KID, I_AO1, I_CA, I_UA,
                                 I_UV, I_SVC, I_IVC, I_RA, I_LA, I_RV, I_LV,
                                 I_P_CS, I_Q_AP, I_BARO_HR_S, I_BARO_HR_V,
                                 I_BARO_R_S, I_BARO_VU_S, I_BARO_E_S,
                                 I_C_CAT, I_EDEMA, I_RAAS, I_AVP,
                                 I_X_AUTO, I_REGURG)
from project_paths import clinical_csv

# ===========================================================================
#  SIMULATION SETTINGS
# ===========================================================================
# Matches optimizer_v1.py's convention so costs stay comparable to prior work.
SIM_DURATION     = 20.0   # [s] length of each run
ANALYSIS_SECONDS = 8.0    # [s] only the LAST this many are scored (drops the
                          #     start-up transient)
MAX_STEP_COARSE  = 0.01   # used for profile scans / optimizer search
MAX_STEP_FINE    = 0.002  # matches AP_model_v17.run_simulation exactly
RTOL, ATOL       = 1e-6, 1e-8
RESAMPLE_HZ      = 200.0
AMP_PCTL         = (2.5, 97.5)
P_INIT           = 16.5   # [mmHg] filling pressure (from v17.run_simulation)

# Scored features.
#
# dp_pump_amp is computed but deliberately NOT scored -- see DIAGNOSTIC_ONLY.
FEATURES = ["q_mean", "q_amp", "dp_pump_mean",
            "dp_ecmo", "dp_restr", "p_pre_ecmo", "p_ua_mean"]

# Computed and reported, but excluded from J.
#
#   dp_pump_amp -- the model's pump is the STATIC curve dPump(Q) = -0.2638 Q +
#   62.207, so the model's pump pressure amplitude is exactly 0.2638 * q_amp:
#   an algebraic restatement of a feature already scored, carrying no
#   independent information about any parameter.  A real centrifugal pump has
#   inertance and its dP responds to dQ/dt as well as Q, so the recording shows
#   1.96 mmHg where the static curve can only ever give 0.18.  Scoring it added
#   an irreducible ~26.5 to every evaluation -- 99.4% of the total cost -- and
#   biased the fit, because the only way to raise the model's value is to drive
#   q_amp to ~446 mL/min against its own clinical target of 40.
#
#   The gap is a statement about MODEL STRUCTURE, not about parameter values,
#   and the honest fix is pump inertance in a future version, not a parameter
#   search.  It is still computed and printed so the deficiency stays visible.
DIAGNOSTIC_ONLY = ["dp_pump_amp"]

ALL_FEATURES = FEATURES + DIAGNOSTIC_ONLY

WEIGHTS = {f: 1.0 for f in FEATURES}

# A caveat that applies to the features that ARE scored, and that any
# interpretation of J should carry:
#
# The AP circuit is a chain of FIXED R-L elements driven by Q_AP and anchored at
# P_UA, so several scored features are near-deterministic functions of q_mean:
#     dp_ecmo     ~ R_ECMO * q_mean      (2.2492 * 3.157 = 7.10; model gives 7.09)
#     dp_restr    ~ R_RES  * q_mean      (0.7476 * 3.157 = 2.36; model gives 2.35)
#     dp_pump_mean = 62.207 - 0.2638 * q_mean
#     p_pre_ecmo   = p_ua_mean + dp_pump_mean
# They are kept because each is an independent MEASUREMENT (a separate clinical
# transducer, with its own noise), so agreeing with all of them is a real check
# on the circuit.  But they are not independent CONSTRAINTS on the parameters:
# the effective information in J is closer to "mean flow, pulse amplitude, and
# UA pressure" than to seven separate numbers, and mean flow is in effect
# weighted several times over.  This is worth remembering when reading how many
# parameters came back non-identifiable -- some of that is the model, and some
# of it is that a 30 s AP-circuit recording simply contains few independent
# numbers.

# A failed integration must be finite (SLSQP's finite differences choke on inf)
# but far worse than any feasible point.
FAIL_COST = 1e6


# ===========================================================================
#  CIRCUIT PARAMETERS  (nuisance parameters, re-optimised inside each profile)
# ===========================================================================
# These are the three constants v16 had baked in and v17 reverted to textbook
# baselines.  v17 as written runs at 418 mL/min against a clinical 188, so they
# must be re-fitted before any mechanism parameter can be meaningfully profiled
# -- otherwise every profile's curvature is dominated by circuit mismatch.
CIRCUIT_PARAMS = ["R_AO4_UA", "V_BLOOD_TOTAL", "C_UA"]

# v17's own baselines, used only as the fallback starting point for the fit.
def v17_circuit_defaults():
    m = ArtificialPlacentaModel()
    return {"R_AO4_UA": m.R["AO4_UA"],
            "V_BLOOD_TOTAL": float(M17.BLOOD_VOLUME),
            "C_UA": m.C["UA"]}


# Wide bounds: v16's working values (40.545 / 268.787 / 0.002471) and v17's
# baselines (3.9*k_R ~ 6.4 / 70 / 0.1*k_C ~ 0.07) are a factor of ~30 apart in
# C_UA, so the box has to contain both or the fit repeats optimizer_v1.py's
# original mistake of excluding the answer.
CIRCUIT_BOUNDS = {
    "R_AO4_UA":      (1.0,    120.0),    # mmHg*s/mL
    "V_BLOOD_TOTAL": (50.0,   600.0),    # mL
    "C_UA":          (5e-4,   0.30),     # mL/mmHg
}


# ===========================================================================
#  MECHANISM PARAMETER REGISTRY
# ===========================================================================
# One entry per profiled parameter:
#   name       attribute on ArtificialPlacentaModel (or a synthetic name below)
#   mech       which of the 12 mechanisms it belongs to (for plot grouping)
#   scale      "log"    -> profiled geometrically, must stay > 0
#              "linear" -> profiled arithmetically (fractions, [0,1] gains)
#   lo, hi     hard bounds; the profile grid spans these
#   label      axis label for the plot grid
#
# `ref` (the reference value) is NOT stored here -- it is read off a live v17
# instance at run time, so this registry can never drift out of sync with the
# model.  The six auto-calibrated parameters (baro_P_set/k, raas_P_half/k,
# avp_P_half/k) and auto_P_ref read `None` on a fresh instance; their reference
# values are the values run_simulation computes, recovered by reference_values().

class Param:
    __slots__ = ("name", "mech", "scale", "lo", "hi", "label", "note")

    def __init__(self, name, mech, scale, lo, hi, label=None, note=""):
        self.name  = name
        self.mech  = mech
        self.scale = scale
        self.lo    = lo
        self.hi    = hi
        self.label = label or name
        self.note  = note

    def grid(self, ref, n):
        """n points spanning [lo, hi], always including `ref` exactly."""
        ref = float(np.clip(ref, self.lo, self.hi))
        if self.scale == "log":
            lo = max(self.lo, 1e-12)
            g = np.geomspace(lo, self.hi, n)
        else:
            g = np.linspace(self.lo, self.hi, n)
        # snap the nearest grid point onto the reference so dJ = 0 there exactly
        g[int(np.argmin(np.abs(g - ref)))] = ref
        return np.unique(g)


# --- Mechanism 1: arterial baroreflex --------------------------------------
BARO = [
    Param("baro_P_set",   "baroreflex", "log",    2.0,   40.0,  r"$P_{set}$ (mmHg)"),
    Param("baro_k",       "baroreflex", "log",    0.05,  10.0,  r"$k_{baro}$"),
    Param("baro_tau_filt","baroreflex", "log",    0.05,   20.0, r"$\tau_{filt}$ (s)"),
    Param("baro_tau_HR_s","baroreflex", "log",    0.1,    40.0, r"$\tau_{HR,symp}$ (s)"),
    Param("baro_tau_HR_v","baroreflex", "log",    0.1,    30.0, r"$\tau_{HR,vagal}$ (s)"),
    Param("baro_tau_R_s", "baroreflex", "log",    0.3,   120.0, r"$\tau_{R,symp}$ (s)"),
    Param("baro_tau_Vu_s","baroreflex", "log",    0.5,   300.0, r"$\tau_{Vu,symp}$ (s)"),
    Param("baro_tau_E_s", "baroreflex", "log",    0.4,   160.0, r"$\tau_{E,symp}$ (s)"),
    Param("baro_G_HR_s",  "baroreflex", "linear", 0.0,    1.0,  r"$G_{HR,symp}$"),
    Param("baro_G_HR_v",  "baroreflex", "linear", 0.0,    1.0,  r"$G_{HR,vagal}$"),
    Param("baro_G_R",     "baroreflex", "linear", 0.0,    1.5,  r"$G_{R}$"),
    Param("baro_G_Vu",    "baroreflex", "linear", 0.0,    1.0,  r"$G_{Vu}$"),
    Param("baro_G_E",     "baroreflex", "linear", 0.0,    1.0,  r"$G_{E}$"),
    Param("HR_intrinsic", "baroreflex", "log",    1.0,    5.0,  r"$HR_{intr}$ (Hz)"),
]

# --- Mechanism 2: Frank-Starling / ventricular mechanics --------------------
FRANK = [
    Param("fs_alpha",      "frank_starling", "linear", 0.0,  2.0,   r"$\alpha_{FS}$"),
    Param("fs_V_d_frac",   "frank_starling", "linear", 0.0,  0.5,   r"$V_d$ frac"),
    Param("fs_V_ref_frac", "frank_starling", "log",    0.3,  8.0,   r"$V_{ref}$ frac"),
    Param("edpvr_A",       "frank_starling", "log",    1e-5, 0.5,   r"EDPVR $A$"),
    Param("edpvr_B",       "frank_starling", "log",    1e-4, 0.5,   r"EDPVR $B$"),
    Param("E_max_L",       "frank_starling", "log",    2.0,  200.0, r"$E_{max,LV}$"),
    Param("E_max_R",       "frank_starling", "log",    2.0,  200.0, r"$E_{max,RV}$"),
    Param("E_min_L",       "frank_starling", "log",    0.02, 5.0,   r"$E_{min,LV}$"),
    Param("E_min_R",       "frank_starling", "log",    0.02, 5.0,   r"$E_{min,RV}$"),
    Param("E_max_LA",      "frank_starling", "log",    0.5,  50.0,  r"$E_{max,LA}$"),
    Param("E_max_RA",      "frank_starling", "log",    0.3,  40.0,  r"$E_{max,RA}$"),
]

# --- Mechanism 3: catecholamine stress surge --------------------------------
CAT = [
    Param("cat_C_peak",   "catecholamine", "log",    1.0,   20.0,     r"$C_{peak}$"),
    Param("cat_tau_fast", "catecholamine", "log",    10.0,  3000.0,   r"$\tau_{fast}$ (s)"),
    Param("cat_tau_slow", "catecholamine", "log",    300.0, 72000.0,  r"$\tau_{slow}$ (s)"),
    Param("cat_w_fast",   "catecholamine", "linear", 0.0,   1.0,      r"$w_{fast}$"),
    Param("cat_G_HR",     "catecholamine", "linear", 0.0,   0.8,      r"$G_{HR,cat}$"),
    Param("cat_G_R",      "catecholamine", "linear", 0.0,   0.8,      r"$G_{R,cat}$"),
    Param("cat_G_E",      "catecholamine", "linear", 0.0,   0.8,      r"$G_{E,cat}$"),
]

# --- Mechanism 4: myocardial edema ------------------------------------------
EDEMA = [
    Param("edema_tau",            "edema", "log",    60.0, 3.6e5, r"$\tau_{edema}$ (s)"),
    Param("edema_max_stiffening", "edema", "linear", 1.0,  6.0,   r"max stiffening"),
]

# --- Mechanism 5: venous unstressed volumes ---------------------------------
# v17 ships every Vu_frac entry at 0.0, which makes Vu identically zero and the
# whole mechanism inert (see compute_derivatives: Vu_base = Vu_frac[key] * ...).
# Profiling 19 individually-zero dict entries is pointless, so one synthetic
# scalar drives all of them; profile_likelihood applies it via set_param().
VENOUS = [
    Param("Vu_frac_global", "venous_Vu", "linear", 0.0, 0.6, r"$V_u$ fraction (all)",
          note="synthetic: sets Vu_frac for every compartment at once"),
]

# --- Mechanism 6: peripheral chemoreflex ------------------------------------
CHEMO = [
    Param("PaO2",            "chemoreflex", "log",    10.0, 400.0, r"$P_aO_2$ (mmHg)"),
    Param("chemo_PaO2_half", "chemoreflex", "log",    2.0,  120.0, r"$P_aO_2^{1/2}$ (mmHg)"),
    Param("chemo_n_hill",    "chemoreflex", "log",    0.5,  10.0,  r"$n_{Hill}$"),
    Param("chemo_G_R",       "chemoreflex", "linear", 0.0,  1.5,   r"$G_{R,chemo}$"),
    Param("chemo_G_HR",      "chemoreflex", "linear", 0.0,  1.0,   r"$G_{HR,chemo}$"),
]

# --- Mechanism 7: cerebral autoregulation -----------------------------------
AUTO = [
    Param("auto_alpha", "cerebral_auto", "linear", 0.0,  1.0,   r"$\alpha_{auto}$"),
    Param("auto_tau",   "cerebral_auto", "log",    0.5, 200.0,  r"$\tau_{auto}$ (s)"),
    Param("auto_P_ref", "cerebral_auto", "log",    1.0, 100.0,  r"$P_{ref,auto}$ (mmHg)"),
]

# --- Mechanism 8: ductus arteriosus reactivity ------------------------------
DA = [
    Param("da_PaO2_half", "ductus", "log",    5.0,  300.0, r"$P_aO_2^{1/2}$ DA (mmHg)"),
    Param("da_k_O2",      "ductus", "log",    0.005, 2.0,  r"$k_{O_2}$"),
    Param("da_G_O2",      "ductus", "linear", 0.0,  6.0,   r"$G_{O_2}$"),
    Param("PGE1_dose",    "ductus", "linear", 0.0,  5.0,   r"PGE$_1$ dose"),
    Param("da_K_PGE",     "ductus", "log",    0.02, 10.0,  r"$K_{PGE}$"),
    Param("da_G_PGE",     "ductus", "linear", 0.0,  1.0,   r"$G_{PGE}$"),
]

# --- Mechanism 9: pulmonary vascular resistance -----------------------------
PVR = [
    Param("pvr_PaO2_ref", "pvr", "log",    5.0,  200.0, r"$P_aO_2^{ref}$ (mmHg)"),
    Param("pvr_beta",     "pvr", "linear", 0.0,  2.0,   r"$\beta_{PVR}$"),
]

# --- Mechanism 10: valve regurgitation --------------------------------------
REGURG = [
    Param("regurg_tau",         "regurgitation", "log",    60.0, 3.6e5, r"$\tau_{regurg}$ (s)"),
    Param("regurg_V_threshold", "regurgitation", "linear", 0.8,  2.5,   r"EDV threshold"),
    Param("regurg_k_growth",    "regurgitation", "log",    1e-7, 1e-1,  r"$k_{growth}$"),
]

# --- Mechanism 11: RAAS ------------------------------------------------------
RAAS = [
    Param("raas_tau",    "raas", "log",    30.0, 3.6e4, r"$\tau_{RAAS}$ (s)"),
    Param("raas_P_half", "raas", "log",    1.0,  60.0,  r"$P_{1/2,RAAS}$ (mmHg)"),
    Param("raas_k",      "raas", "log",    0.02, 10.0,  r"$k_{RAAS}$"),
    Param("raas_G_R",    "raas", "linear", 0.0,  1.5,   r"$G_{R,RAAS}$"),
]

# --- Mechanism 12: vasopressin ----------------------------------------------
AVP = [
    Param("avp_tau",    "vasopressin", "log",    30.0, 3.6e4, r"$\tau_{AVP}$ (s)"),
    Param("avp_P_half", "vasopressin", "log",    1.0,  60.0,  r"$P_{1/2,AVP}$ (mmHg)"),
    Param("avp_k",      "vasopressin", "log",    0.02, 10.0,  r"$k_{AVP}$"),
    Param("avp_G_R",    "vasopressin", "linear", 0.0,  1.5,   r"$G_{R,AVP}$"),
]

MECH_PARAMS = (BARO + FRANK + CAT + EDEMA + VENOUS + CHEMO + AUTO
               + DA + PVR + REGURG + RAAS + AVP)
MECH_BY_NAME = {p.name: p for p in MECH_PARAMS}

# Mechanism -> the toggle attribute that must be True for it to have any effect.
MECH_TOGGLE = {
    "baroreflex":     "baroreflex",
    "frank_starling": "frank_starling",
    "catecholamine":  "catecholamine_stress",
    "edema":          "myocardial_edema",
    "venous_Vu":      "venous_unstressed_vol",
    "chemoreflex":    "chemoreflex",
    "cerebral_auto":  "cerebral_autoregulation",
    "ductus":         "da_reactivity",
    "pvr":            "pvr_modulation",
    "regurgitation":  "valve_regurgitation",
    "raas":           "raas",
    "vasopressin":    "vasopressin",
}

MECH_ORDER = ["baroreflex", "frank_starling", "catecholamine", "edema",
              "venous_Vu", "chemoreflex", "cerebral_auto", "ductus",
              "pvr", "regurgitation", "raas", "vasopressin"]


def set_param(model, name, value):
    """Apply one profiled parameter to a live model instance."""
    if name == "Vu_frac_global":
        for k in model.Vu_frac:
            model.Vu_frac[k] = float(value)
    elif name == "R_AO4_UA":
        model.R["AO4_UA"] = float(value)
        model.R_baseline["AO4_UA"] = float(value)
    elif name == "C_UA":
        model.C["UA"] = float(value)
        model.c_list = [model.C[k] for k in model._c_keys]
    else:
        setattr(model, name, float(value))


# ===========================================================================
#  MODEL CONSTRUCTION AND SIMULATION
# ===========================================================================
def build_model(overrides=None, toggles=None):
    """A fresh v17 instance with `overrides` (name -> value) applied.

    Always builds a NEW instance: v17 caches auto-calibrated set-points on
    `self` (`_baro_P_set_cal`, `auto_P_ref`, ...), so reusing an instance would
    leak the previous evaluation's calibration into the next one.
    """
    model = ArtificialPlacentaModel(toggles=toggles)
    for name, value in (overrides or {}).items():
        if name == "V_BLOOD_TOTAL":
            continue                     # handled in make_x0, not on the model
        set_param(model, name, value)
    return model


def make_x0(model, V_blood_total):
    """Initial state + set-point auto-calibration.

    This is a faithful transcription of AP_model_v17.run_simulation's setup with
    two changes: the total blood volume is an argument instead of the module
    constant, and the solver call is left to `simulate` so max_step can be
    varied.  `verify_against_v17()` at the bottom of this file asserts the
    transcription still matches the model bit for bit.
    """
    x0 = np.zeros(N_STATES)
    for i, c in enumerate(model.c_list):
        x0[i] = c * P_INIT
    x0[I_RA], x0[I_LA], x0[I_RV], x0[I_LV] = 4.0, 2.0, 10.4, 6.5
    x0[0:23] *= float(V_blood_total) / x0[0:23].sum()

    model._V_init    = x0[0:19].copy()
    model._V_RV_init = x0[I_RV]
    model._V_LV_init = x0[I_LV]

    P_AR_init  = x0[I_AR]  / model.C['AR']
    P_KID_init = x0[I_KID] / model.C['KID']
    P_AO1_init = x0[I_AO1] / model.C['AO1']
    P_CA_init  = x0[I_CA]  / model.C['CA']

    x0[I_P_CS] = P_AR_init
    x0[I_Q_AP] = model.steady_state_flow(x0[I_UA] / model.C['UA'],
                                         x0[I_UV] / model.C['UV'])
    x0[I_BARO_HR_S] = 0.5
    x0[I_BARO_HR_V] = 0.5
    x0[I_BARO_R_S]  = 0.5
    x0[I_BARO_VU_S] = 0.5
    x0[I_BARO_E_S]  = 0.5
    x0[I_C_CAT]  = model.cat_C_peak if model.flags.catecholamine_stress else 1.0
    x0[I_EDEMA]  = 0.0
    x0[I_RAAS]   = 0.0
    x0[I_AVP]    = 0.0
    x0[I_X_AUTO] = model.R['CA_BR']
    x0[I_REGURG] = 0.0

    model._baro_P_set_cal = (model.baro_P_set if model.baro_P_set is not None
                             else P_AR_init)
    model._baro_k_cal = (model.baro_k if model.baro_k is not None
                         else 2.2 / (0.30 * max(1e-3, model._baro_P_set_cal)))
    model._raas_P_half_cal = (model.raas_P_half if model.raas_P_half is not None
                              else P_KID_init)
    model._raas_k_cal = (model.raas_k if model.raas_k is not None
                         else 2.2 / (0.30 * max(1e-3, P_KID_init)))
    model._avp_P_half_cal = (model.avp_P_half if model.avp_P_half is not None
                             else P_AR_init)
    model._avp_k_cal = (model.avp_k if model.avp_k is not None
                        else 2.2 / (0.30 * max(1e-3, P_AR_init)))
    if model.auto_P_ref is None:
        model.auto_P_ref = (P_AO1_init + P_CA_init) / 2.0
    return x0


def simulate(overrides, max_step=MAX_STEP_COARSE, duration=SIM_DURATION,
             toggles=None):
    """Run v17 with `overrides` applied.  Returns (sol, model) or (None, model)."""
    ov = dict(overrides or {})
    V_blood = ov.get("V_BLOOD_TOTAL", float(M17.BLOOD_VOLUME))
    model = build_model(ov, toggles=toggles)
    x0 = make_x0(model, V_blood)
    try:
        sol = solve_ivp(model.compute_derivatives, [0.0, duration], x0,
                        method="Radau", max_step=max_step, rtol=RTOL, atol=ATOL)
    except Exception:
        return None, model
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        return None, model
    return sol, model


# ===========================================================================
#  FEATURES
# ===========================================================================
def _amp(x):
    return float(np.percentile(x, AMP_PCTL[1]) - np.percentile(x, AMP_PCTL[0]))


def model_features(sol, model):
    """The 8 scored features, time-weighted on a uniform grid."""
    t_cut = sol.t[-1] - ANALYSIS_SECONDS
    sel = sol.t >= t_cut
    if sel.sum() < 10:
        sel = np.ones_like(sol.t, dtype=bool)
    t = sol.t[sel]

    q_raw    = sol.y[I_Q_AP, sel] * 60.0                  # mL/min
    P_UA_raw = sol.y[I_UA, sel] / model.C['UA']
    P_UV_raw = sol.y[I_UV, sel] / model.C['UV']

    n = max(64, int(round((t[-1] - t[0]) * RESAMPLE_HZ)))
    t_u  = np.linspace(t[0], t[-1], n)
    q    = np.interp(t_u, t, q_raw)
    P_UA = np.interp(t_u, t, P_UA_raw)
    P_UV = np.interp(t_u, t, P_UV_raw)

    _, pump, pre_ecmo, post_ecmo, post_res = model.ap_loop(P_UA, P_UV, q / 60.0)

    return {
        "q_mean":       float(q.mean()),
        "q_amp":        _amp(q),
        "dp_pump_mean": float(pump.mean()),
        "dp_pump_amp":  _amp(pump),          # diagnostic only, not scored
        "dp_ecmo":      float(np.mean(pre_ecmo - post_ecmo)),
        "dp_restr":     float(np.mean(post_ecmo - post_res)),
        "p_pre_ecmo":   float(pre_ecmo.mean()),
        "p_ua_mean":    float(P_UA.mean()),
    }


_CLIN_CACHE = {}


def clinical_targets(csv_path=None):
    """Clinical feature values c_f and their noise scales sigma_f.

    sigma_f for a MEAN feature is the within-window SD of the signal; for an
    AMPLITUDE feature it is the SD of per-second amplitudes (how much the pulse
    size itself varies beat to beat).  A floor of 5% of |c_f| keeps a very
    steady channel from dominating the whole cost.
    """
    path = csv_path or clinical_csv()
    if path in _CLIN_CACHE:
        return _CLIN_CACHE[path]

    clin = ClinicalData(path)
    mask = clin.get_time_window_indices(START_TIME, END_TIME)
    if mask.sum() == 0:
        raise ValueError(f"No clinical rows between {START_TIME} and {END_TIME}.")

    t        = clin._times[mask]
    q        = clin._flow_ml_s[mask] * 60.0
    dp_pump  = clin._dp_pump[mask]
    pre_e    = clin._p_pre_ecmo[mask]
    post_e   = clin._p_post_ecmo[mask]
    post_r   = clin._p_post_res[mask]
    p_ua     = clin._p_pre_pump[mask]

    def amp_sd(sig):
        """SD of the amplitude measured in consecutive 1-second blocks."""
        n_per = int(round(ClinicalData.SAMPLE_RATE_HZ))
        blocks = [sig[i:i + n_per] for i in range(0, len(sig) - n_per + 1, n_per)]
        amps = [_amp(b) for b in blocks if len(b) == n_per]
        return float(np.std(amps)) if len(amps) > 1 else float(np.std(sig))

    c = {
        "q_mean":       float(q.mean()),
        "q_amp":        _amp(q),
        "dp_pump_mean": float(dp_pump.mean()),
        "dp_pump_amp":  _amp(dp_pump),
        "dp_ecmo":      float(np.mean(pre_e - post_e)),
        "dp_restr":     float(np.mean(post_e - post_r)),
        "p_pre_ecmo":   float(pre_e.mean()),
        "p_ua_mean":    float(p_ua.mean()),
    }
    s = {
        "q_mean":       float(q.std()),
        "q_amp":        amp_sd(q),
        "dp_pump_mean": float(dp_pump.std()),
        "dp_pump_amp":  amp_sd(dp_pump),
        "dp_ecmo":      float(np.std(pre_e - post_e)),
        "dp_restr":     float(np.std(post_e - post_r)),
        "p_pre_ecmo":   float(pre_e.std()),
        "p_ua_mean":    float(p_ua.std()),
    }
    for k in ALL_FEATURES:
        s[k] = max(s[k], 0.05 * abs(c[k]), 1e-6)

    _CLIN_CACHE[path] = (c, s, float(t[-1] - t[0]), int(mask.sum()))
    return _CLIN_CACHE[path]


# ===========================================================================
#  THE OBJECTIVE
# ===========================================================================
def cost_from_features(m_f, c_f, s_f):
    terms = {k: WEIGHTS[k] * ((m_f[k] - c_f[k]) / s_f[k]) ** 2 for k in FEATURES}
    return float(sum(terms.values())), terms


def evaluate(overrides, max_step=MAX_STEP_COARSE, duration=SIM_DURATION,
             toggles=None, want_terms=False):
    """J(theta) for one parameter set.  Returns cost, or (cost, terms)."""
    c_f, s_f, _, _ = clinical_targets()
    with contextlib.redirect_stdout(io.StringIO()):
        sol, model = simulate(overrides, max_step=max_step, duration=duration,
                              toggles=toggles)
    if sol is None:
        return (FAIL_COST, {k: FAIL_COST / len(FEATURES) for k in FEATURES}) \
            if want_terms else FAIL_COST
    try:
        m_f = model_features(sol, model)
    except Exception:
        return (FAIL_COST, {k: FAIL_COST / len(FEATURES) for k in FEATURES}) \
            if want_terms else FAIL_COST
    if not all(np.isfinite(v) for v in m_f.values()):
        return (FAIL_COST, {k: FAIL_COST / len(FEATURES) for k in FEATURES}) \
            if want_terms else FAIL_COST
    cost, terms = cost_from_features(m_f, c_f, s_f)
    cost = min(cost, FAIL_COST)
    return (cost, terms) if want_terms else cost


# ===========================================================================
#  REFERENCE VALUES
# ===========================================================================
def reference_values(circuit=None):
    """Reference (nominal) value of every profiled parameter.

    Mechanism parameters are read off a live v17 instance; the six auto-
    calibrated ones (which read None) are recovered from the values
    make_x0 computes, so they are profiled around what the model ACTUALLY uses
    rather than around a placeholder.
    """
    circ = dict(v17_circuit_defaults())
    if circuit:
        circ.update(circuit)
    model = build_model(circ)
    with contextlib.redirect_stdout(io.StringIO()):
        make_x0(model, circ["V_BLOOD_TOTAL"])

    calibrated = {
        "baro_P_set":  model._baro_P_set_cal,
        "baro_k":      model._baro_k_cal,
        "raas_P_half": model._raas_P_half_cal,
        "raas_k":      model._raas_k_cal,
        "avp_P_half":  model._avp_P_half_cal,
        "avp_k":       model._avp_k_cal,
        "auto_P_ref":  model.auto_P_ref,
    }

    ref = {}
    for p in MECH_PARAMS:
        if p.name in calibrated:
            ref[p.name] = float(calibrated[p.name])
        elif p.name == "Vu_frac_global":
            ref[p.name] = float(model.Vu_frac['BR'])
        else:
            ref[p.name] = float(getattr(model, p.name))
    ref.update(circ)
    return ref


# ===========================================================================
#  SELF-TEST
# ===========================================================================
def verify_against_v17(tol=1e-9):
    """Assert make_x0 + solve_ivp reproduces v17.run_simulation exactly.

    make_x0 duplicates run_simulation's initialisation so that blood volume and
    max_step can be varied.  If v17 ever changes, this catches the drift instead
    of letting the analyses silently score a different model.
    """
    m_ref = ArtificialPlacentaModel()
    with contextlib.redirect_stdout(io.StringIO()):
        sol_ref = m_ref.run_simulation(duration=4.0)
        m_new = build_model({})
        x0 = make_x0(m_new, float(M17.BLOOD_VOLUME))
        sol_new = solve_ivp(m_new.compute_derivatives, [0.0, 4.0], x0,
                            method="Radau", max_step=MAX_STEP_FINE,
                            rtol=RTOL, atol=ATOL)
    d = np.max(np.abs(sol_ref.y[:, -1] - sol_new.y[:, -1]))
    ok = d < tol
    return ok, float(d)


if __name__ == "__main__":
    ok, d = verify_against_v17()
    print(f"transcription self-test: {'PASS' if ok else 'FAIL'}  (max |dx| = {d:.3e})")

    c_f, s_f, win, n = clinical_targets()
    print(f"\nclinical window {START_TIME}-{END_TIME}: {win:.1f} s, {n} samples")
    print(f"{'feature':>14} {'clinical':>12} {'sigma':>10}")
    for k in ALL_FEATURES:
        tag = "  (diagnostic only)" if k in DIAGNOSTIC_ONLY else ""
        print(f"{k:>14} {c_f[k]:>12.3f} {s_f[k]:>10.3f}{tag}")

    print(f"\nmechanism parameters registered: {len(MECH_PARAMS)}")
    cost, terms = evaluate({}, max_step=MAX_STEP_FINE, want_terms=True)
    print(f"\nJ(v17 as written) = {cost:.1f}")
    for k in FEATURES:
        print(f"   {k:>14}: {terms[k]:>12.2f}")
