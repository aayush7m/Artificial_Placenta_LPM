# Clinical data (not included)

The ECMO trial recordings this project was fitted against are **not distributed
with this repository**. To run the models and analysis scripts, place a recording
in this folder in the format below, or point the scripts at one elsewhere with the
`ECMO_CSV_PATH` environment variable.

## Expected files

| File | Used by |
|---|---|
| `ECMO_Trial_2024-03-05_cleaned_v2.csv` | **Default** (`project_paths.DEFAULT_CSV`). Every current model and analysis scores against this. |
| `ECMO_Trial_2024-03-05_cleaned_v1.csv` | Superseded earlier cleaning of the same 2024-03-05 trial. |
| `ECMO_short_trial.csv` | Early short extract, used only by `superseded/AP_model_v6.py`. |

## Format

UTF-8 CSV (a BOM is fine), exported from the bedside acquisition system,
**sampled at 10 Hz**. The loaders (`ClinicalData` in each model) skip the first
**two** rows and then read columns **by position**, not by name — so the column
order matters and the header text does not.

- **Row 1:** column names
- **Row 2:** units
- **Row 3 onward:** one sample per row

| Index | Column | Units | Read by the models? |
|---|---|---|---|
| 0 | Flow (AP circuit) | mL/min | **yes** |
| 1 | Temperature | °C | no |
| 2 | Pre-ECMO pressure | mmHg | **yes** |
| 3 | Post-ECMO pressure | mmHg | **yes** |
| 4 | Post-restrictor pressure | mmHg | **yes** |
| 5 | Pre-pump pressure | mmHg | **yes** |
| 6 | Jugular vein pressure | mmHg | no |
| 7 | HR from carotid pressure | BPM | no |
| 8 | HR from flow | BPM | no |
| 9 | Wall-clock timestamp, `HH:MM:SS` | — | **yes** (to locate the analysis window) |
| 10 | Event annotation (free text) | — | no |

Example of the first three rows:

```
Flow,Temperature,Pre-ECMO Pressure,Post-ECMO Pressure ,Post-Restrictor Pressure,Pre-Pump Pressure,Jugular Vein Pressure #2,HR from Carotid Pressure,HR from FLOW, ,Channel 14
mL/min,°C,mmHg,mmHg,mmHg,mmHg,mmHg,BPM,BPM, , 
<flow>,<temp>,<p_pre_ecmo>,<p_post_ecmo>,<p_post_restrictor>,<p_pre_pump>,<p_jugular>,<hr_carotid>,<hr_flow>,13:08:05,<note>
```

Rows with fewer than 6 fields are skipped; a non-numeric value in columns 0 or 2–5
aborts the load. Elapsed time is taken from the row index (`index / 10 Hz`), and the
pump pressure rise is computed as `pre-ECMO − pre-pump`.

The analysis window used throughout is `START_TIME = 13:38:24` → `END_TIME = 13:38:54`
(30 s), matched against column 9. A different recording needs those constants
changed at the top of the model / analysis script.
