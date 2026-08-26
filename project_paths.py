"""
Canonical project paths, defined in one place.

Every runnable script in this project starts with the same four-line bootstrap
so it can be launched directly (``python3 analysis/optimizer_v1.py``) from
anywhere, without needing ``python3 -m`` or a PYTHONPATH export:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from project_paths import DEFAULT_CSV, RESULTS_DIR

After that bootstrap, models are imported as a package:

    from models.AP_model_v16 import ArtificialPlacentaModel

Every data and results path below is absolute, so scripts behave identically
regardless of the working directory they are launched from.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

# The clinical recording every current model and analysis is scored against.
DEFAULT_CSV = os.path.join(DATA_DIR, "ECMO_Trial_2024-03-05_cleaned_v2.csv")

# Per-tool results directories (created on demand by the tools themselves).
RESULTS_AP_RL_FIT = os.path.join(RESULTS_DIR, "ap_rl_fit")
RESULTS_OPTIMIZER = os.path.join(RESULTS_DIR, "optimizer")
RESULTS_PROFILE_V16 = os.path.join(RESULTS_DIR, "profile_likelihood_v16")
RESULTS_PROFILE_V17 = os.path.join(RESULTS_DIR, "profile_likelihood_v17")
RESULTS_SENSITIVITY_V17 = os.path.join(RESULTS_DIR, "sensitivity_v17")


def clinical_csv():
    """The clinical CSV, overridable with the ECMO_CSV_PATH environment variable."""
    return os.environ.get("ECMO_CSV_PATH", DEFAULT_CSV)


def ensure(directory):
    """Create `directory` if needed and return it."""
    os.makedirs(directory, exist_ok=True)
    return directory
