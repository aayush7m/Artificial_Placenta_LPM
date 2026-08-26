# Abandoned: pre-versioned prototypes

Everything here predates the `AP_model_vN` series, which starts at v4 in
`superseded/`. Two independent attempts:

## Attempt A — `one.py` → `two.py` → `three.py`

Successive from-scratch fetal circulation models (May 26 – Jun 8).

- `one.py` — `FetalCirculationModel`, Pennati (1997) Table 1 parameters, isovolumic
  pressure-generator heart.
- `two.py` — switches to a time-varying **elastance** heart (Yigit supplementary
  data). These `E_max`/`E_min` values survive into v16.
- `three.py` — `AdvancedFetalCirculationModel`, adds the baroreflex (`reflex_on`).
  This is the direct precursor of v4.

Attempt A was not so much abandoned as **renamed and continued** — v4 picks up where
`three.py` stops. It lives here rather than in `superseded/` because the files are
unversioned and were never wired to the clinical data loader.

## Attempt B — `main.py` + `params.py`

A separate radius-based formulation (`get_r_LV`, `get_r_RV`, … deriving chamber radii
from volumes), with parameters split into a `params.py` module. Reduced from 18 to 14
state variables by removing the 4 radius states. Abandoned in favour of the
elastance-based approach; nothing from it carries into the current models.

`params.py` is imported by `main.py` as `old.params`, which no longer resolves after
the reorganization. Both are frozen.
