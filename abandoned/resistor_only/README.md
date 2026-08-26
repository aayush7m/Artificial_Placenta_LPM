# Abandoned: resistor-only AP circuit

## What this was

Each passive AP component (ECMO, restrictor, tubing) modelled as a **pure resistor**,
with R obtained by regressing the measured pressure drop against flow.

- `resistor_fit.py` (Jun 17) — first version.
- `resistor_fit_v2.py` (Jun 22) — adds z-score outlier removal
  (`OUTLIER_REMOVAL_FRACTION = 0.05`).

## Why it was abandoned

A pure resistor has one degree of freedom, which sets the **mean** flow. It cannot
also reproduce the flow **amplitude** — with R fixed to match the mean, the predicted
waveform is far too pulsatile, because nothing attenuates the pulsatile component.

Each component was remodelled as a series R–L element:

```
dP(t) = R·Q(t) + L·dQ/dt        =>        Q(s)/dP(s) = 1/(L·s + R)
```

giving two degrees of freedom: **R sets the mean** (DC gain 1/R) and **L sets the
amplitude**. `analysis/inductor_resistor_fit.py` (Jul 22) replaced both scripts,
and `AP_model_v13` was the first model to use its output.

## Note

These read the CSV with a bare relative path (`pd.read_csv('ECMO_Trial_...csv')`),
so they only ever worked when run from the folder containing the data. They now
require `data/` to be the working directory, or an edit. They also need `pandas`.
