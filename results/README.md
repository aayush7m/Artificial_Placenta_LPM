# Generated results

Everything here is reproducible output — safe to delete and regenerate, though the
profile-likelihood cache represents ~5 CPU-hours.

| Folder | Produced by | Contents |
|---|---|---|
| `ap_rl_fit/` | `analysis/inductor_resistor_fit.py` | `ap_rl_fit_results.json` — the fitted AP-circuit R–L pairs. **These values are hand-copied into the `AP_RL` table in v13–v16**, so regenerating this file does not update the models. |
| `optimizer/` | `analysis/optimizer_v1.py` | `optimizer_state.json` (not committed; created on first run) — resumable best-so-far for the 3 circuit parameters. `archive/` keeps the pre-bounds-fix state. Note: does **not** match the constants in v15/v16 (see top-level README). |
| `profile_likelihood_v17/` | `analysis/profile_likelihood_v17.py` | `profile_state.json` — resumable cache of the circuit fit, the 62-parameter sensitivity scan, and every profile. `profile_likelihood_grid.png` — all parabolas in one grid. `parameter_ranking.png`, `profile_summary.csv` — the identifiability verdicts and 95% confidence intervals. Represents ~5 CPU-hours. |
| `sensitivity_v17/` | `analysis/sensitivity_v17.py` | `fitted_parameters.json` — the fitted values **and** their provenance; this is the file `analysis/make_v18.py` reads to generate `models/AP_model_v18.py`. `sensitivity_grid.png` — local sensitivity heatmap, tornado chart, collinearity matrix. `fit_comparison.png` — traces and per-feature cost, v17 vs circuit-refit vs full fit. |

## The v17 → v18 pipeline

```
data/ECMO_Trial_2024-03-05_cleaned_v2.csv
              │
              ▼
    analysis/ap_cost.py            chi-square cost J over 8 phase-invariant features
              │
              ▼
 analysis/profile_likelihood_v17.py
     A. re-fit the 3 circuit constants v17 had reverted
     B. scan all 62 mechanism parameters
     C. profile the ones the scan leaves in doubt
              │  results/profile_likelihood_v17/
              ▼
   analysis/sensitivity_v17.py     joint fit of the identifiable parameters only
              │  results/sensitivity_v17/fitted_parameters.json
              ▼
    analysis/make_v18.py           generates models/AP_model_v18.py
```

Unlike the `ap_rl_fit` values, the v18 constants are **not** hand-copied: `make_v18.py`
reads `fitted_parameters.json` directly, so the model and its provenance record cannot
drift apart. Re-running the pipeline and re-running `make_v18.py` regenerates the model.
