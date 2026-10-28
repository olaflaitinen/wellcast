# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-28

First public release.

### Added

- `wellcast.units`: unit registry and `convert` for pressure, temperature,
  volume, volume rate, gas-liquid ratio, length, time and ratio units, with
  aliases and `register_unit`.
- `wellcast.io`: `read_production` for CSV and Excel files, `from_dataframe`
  for in-memory tables, a canonical production schema with a built-in alias
  map and unit conversion on read, and `read_las` for LAS well logs.
- `wellcast.clean`: shut-in detection, Hampel, rolling z-score and IQR outlier
  flags (per well), regular resampling, limited gap filling and `qc_report`.
- `wellcast.datasets`: `make_synthetic_field`, a physically consistent
  synthetic oil field with injectors, shut-ins, water breakthrough, injected
  anomalies and ground-truth labels.
- `wellcast.pvt`: Standing, Vasquez-Beggs and Beggs-Robinson black-oil
  correlations, Hall-Yarborough gas compressibility, gas formation volume
  factor, black-oil tables and range warnings.
- `wellcast.ipr`: productivity index, linear, Vogel, Fetkovich (with fitting)
  and composite IPRs, IPR curves, and Gilbert, Ros, Baxendell and Achong
  choke equations.
- `wellcast.dca`: Arps exponential, hyperbolic and harmonic models, bounded
  robust fitting, AIC model selection, `DeclineModel` with forecast, EUR,
  remaining reserves, bootstrap P90/P50/P10 and plots, and `fit_many`.
- `wellcast.features`: physics-informed features and leakage-safe lag and
  rolling features per well.
- `wellcast.ml`: `Predictor` for virtual flow metering with LightGBM,
  histogram gradient boosting, random forest and ridge models, time-ordered
  and leave-one-well-out validation, per-well metrics, quantile prediction
  intervals, SHAP explanations and joblib persistence.
- `wellcast.diagnostics`: Chan plot data and water mechanism screening, Hall
  plot with slope-change detection, Havlena-Odeh material balance with OOIP
  fit, and rule-based plus Isolation Forest anomaly detection.
- `wellcast.plotting`: a consistent plotly theme and charts for production
  history, decline curves, IPR, predictions, Chan and Hall plots, anomalies
  and feature importance.
- `wellcast` command-line interface with `qc`, `dca`, `demo` and `app`.
- Multipage Streamlit app with demo data, column mapping, QC, prediction,
  decline, diagnostics and HTML report export.
- Tutorial notebooks, runnable examples and a documentation site.

[0.1.0]: https://github.com/olaflaitinen/wellcast/releases/tag/v0.1.0
