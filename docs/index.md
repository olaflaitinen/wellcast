---
title: Home
hide:
  - navigation
---

![wellcast](assets/banner.svg#only-light)
![wellcast](assets/banner-dark.svg#only-dark)

**Oil and gas well performance analysis for Python:** field data ingestion and
quality control, decline curve analysis, PVT, inflow performance, well and
reservoir diagnostics, machine learning for virtual flow metering, and an
interactive Streamlit app.

## Why wellcast

Production engineers repeat the same steps for every field: clean daily
allocation data, fit decline curves, check inflow performance, look for water
and injectivity problems, and increasingly train models on sensor data. These
steps live in spreadsheets and one-off scripts. wellcast puts them in one
tested, typed library with consistent units, documented correlations and
honest validation.

## What is inside

| Module | Purpose |
|---|---|
| [`units`](user-guide/units.md) | Oilfield unit conversions with a typed registry |
| [`io`](user-guide/io.md) | CSV, Excel and LAS readers into one canonical schema |
| [`clean`](user-guide/clean.md) | Shut-in detection, outlier flags, gap filling, QC report |
| [`datasets`](user-guide/datasets.md) | Synthetic field with ground-truth labels |
| [`pvt`](user-guide/pvt.md) | Standing, Vasquez-Beggs, Beggs-Robinson, Hall-Yarborough |
| [`ipr`](user-guide/ipr.md) | Productivity index, Vogel, Fetkovich, composite IPR, chokes |
| [`dca`](user-guide/dca.md) | Arps decline with AIC selection, bootstrap P90/P50/P10, EUR |
| [`features`](user-guide/features.md) | Physics-informed and leakage-safe features |
| [`ml`](user-guide/ml.md) | Virtual flow metering with validation, intervals and SHAP |
| [`diagnostics`](user-guide/diagnostics.md) | Chan and Hall plots, material balance, anomalies |
| [`plotting`](user-guide/plotting.md) | Consistent interactive plotly charts |

## Sixty seconds

```python
import wellcast

field = wellcast.make_synthetic_field(n_wells=5, years=5, seed=42)
rates = field.production.query("well == 'P-01'")["oil_rate"]

model = wellcast.DeclineModel().fit(rates).bootstrap(200)
print(model)
print(model.eur_quantiles(econ_limit=20.0))
model.plot(rates, econ_limit=20.0).show()
```

Continue with the [installation guide](installation.md), the
[quickstart](quickstart.md) or the [tutorials](tutorials/01_quickstart.ipynb).
