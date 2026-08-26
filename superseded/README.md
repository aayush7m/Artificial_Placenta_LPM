# Superseded models (v4 – v13)

Earlier versions of the `AP_model_vN` series. These are **not dead ends** — each
one is a working ancestor of the current `models/AP_model_v16.py`, kept so the
lineage stays auditable. See the top-level `README.md` for the full version table.

## Landmarks in this folder

- **v6** — first version with the `ClinicalData` CSV loader.
- **v7** — pump curve added with the **wrong sign** (`+0.2374·Q + 50.315`).
- **v8** — pump curve corrected to `-0.2638·Q + 62.207`; unchanged in every version since.
- **v12_1 … v12_6** — six *parallel variants*, not a sequence. v12_6 is the one that
  survived and is still imported by `analysis/inductor_resistor_fit.py`.
- **v13** — the inductor switch: first model to carry the `AP_RL` table.

## Still a live dependency

`superseded/AP_model_v12_6.py` is imported by `analysis/inductor_resistor_fit.py`,
which is pinned to it on purpose: v12_6 is the model the published AP-circuit R–L
values were fitted against, and re-pointing the fit at a newer model would silently
change those constants. **Do not move or edit v12_6** without re-running that fit.

## Running these

Frozen — data paths were not updated during the reorganization. Pass the CSV explicitly:

```bash
ECMO_CSV_PATH="$PWD/data/ECMO_Trial_2024-03-05_cleaned_v2.csv" python3 superseded/AP_model_v13.py
```

(v6 expects `data/ECMO_short_trial.csv` instead.)
