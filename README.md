<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/olaflaitinen/wellcast/main/assets/banner-dark.svg">
    <img alt="wellcast" src="https://raw.githubusercontent.com/olaflaitinen/wellcast/main/assets/banner.svg" width="720">
  </picture>
</p>

<p align="center"><strong>Oil and gas well performance analysis for Python.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/wellcast/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/wellcast?cacheSeconds=3600"></a>
  <a href="https://pypi.org/project/wellcast/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/wellcast?cacheSeconds=3600"></a>
  <a href="https://github.com/olaflaitinen/wellcast/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/olaflaitinen/wellcast/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/olaflaitinen/wellcast"><img alt="Coverage" src="https://codecov.io/gh/olaflaitinen/wellcast/branch/main/graph/badge.svg"></a>
  <a href="https://olaflaitinen.github.io/wellcast"><img alt="Docs" src="https://github.com/olaflaitinen/wellcast/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://github.com/olaflaitinen/wellcast/blob/main/LICENSE"><img alt="License MPL-2.0" src="https://img.shields.io/badge/license-MPL--2.0-blue"></a>
  <a href="https://pepy.tech/project/wellcast"><img alt="Downloads" src="https://static.pepy.tech/badge/wellcast/month"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-261230"></a>
  <a href="https://mypy-lang.org/"><img alt="Typed: mypy" src="https://img.shields.io/badge/typed-mypy%20strict-2a6db2"></a>
  <a href="https://orcid.org/0009-0006-5184-0810"><img alt="ORCID" src="https://img.shields.io/badge/ORCID-0009--0006--5184--0810-a6ce39"></a>
</p>

Production engineers repeat the same analysis for every field: clean daily
allocation data, fit decline curves, check inflow performance, look for water
and injectivity problems, and increasingly train models on sensor data. That
work usually lives in spreadsheets and one-off scripts. **wellcast** puts it in
one tested, typed Python library with consistent oilfield units, documented
correlations with their valid ranges, honest validation of machine learning
models, and an interactive app for engineers who prefer not to write code.

## Highlights

- **Field data in, one schema out.** CSV, Excel and LAS readers map common
  database column names (such as `BORE_OIL_VOL` or `AVG_WHP_P`) and units to
  one canonical layout.
- **Quality control that flags, never deletes.** Shut-ins, spikes, frozen
  gauges, gaps and physically impossible values, per well.
- **Decline curve analysis with uncertainty.** Arps models chosen by AIC,
  robust log-space fitting, bootstrap P90/P50/P10 rates and EUR, batch fits
  for a whole field.
- **Textbook PVT and inflow performance.** Standing, Vasquez-Beggs,
  Beggs-Robinson, Hall-Yarborough, Vogel, Fetkovich, composite IPR and choke
  equations, validated against published values.
- **Virtual flow metering.** Leakage-safe features, LightGBM by default,
  time-ordered and leave-one-well-out validation, quantile intervals and
  SHAP explanations.
- **Well and reservoir diagnostics.** Chan and Hall plots, Havlena-Odeh
  material balance, rule-based and Isolation Forest anomaly detection.
- **Reproducible by design.** A synthetic field with ground-truth labels
  powers every example, test, tutorial and the app demo. No third-party data
  is bundled.
- **Typed and tested.** `mypy --strict`, more than 350 tests with over 99
  percent coverage, doctests on every public example.

## Installation

```bash
pip install wellcast
```

The core needs only NumPy, pandas and SciPy. Optional features come as extras:

| Extra | Installs | Enables |
|---|---|---|
| `io` | lasio, openpyxl | LAS logs, Excel files |
| `ml` | scikit-learn, LightGBM, SHAP, joblib | virtual flow metering, Isolation Forest |
| `viz` | plotly | interactive charts |
| `app` | Streamlit, plotly | `wellcast app` |
| `all` | everything above | all features |

```bash
pip install "wellcast[all]"
```

## 60-second quickstart

```python
import wellcast

field = wellcast.make_synthetic_field(n_wells=5, years=5, seed=42)
production = field.production

print(wellcast.qc_report(production).query("impossible > 0"))

rates = production.query("well == 'P-01'")["oil_rate"]
model = wellcast.DeclineModel().fit(rates).bootstrap(200)
print(model)
print(model.eur_quantiles(econ_limit=20.0))
print(wellcast.fit_many(production, econ_limit=20.0)[["model", "eur", "status"]])
```

## Feature tour

**Units and data input**

```python
from wellcast import convert, read_production

convert(250.0, "bar", "psi")  # 3625.9
df = read_production("export.xlsx", units={"oil_rate": "sm3/d", "whp": "bar"})
```

**Cleaning and quality control**

```python
from wellcast.clean import detect_shut_ins, flag_outliers, qc_report

shut_in = detect_shut_ins(df)
spikes = flag_outliers(df, "oil_rate", method="hampel")
report = qc_report(df)
```

**PVT**

```python
from wellcast import pvt

pvt.bubble_point_standing(
    rs=751.0, gas_gravity=0.851, api=47.1, temperature=250.0
)  # 2181.3 psia
pvt.z_factor(pressure=2000.0, temperature=180.0, gas_gravity=0.7)  # 0.844
table = pvt.black_oil_table([1000.0, 2000.0, 3000.0], 200.0, 35.0, 0.75, 500.0)
```

**Inflow performance and chokes**

```python
from wellcast import ipr

qmax = ipr.vogel_qmax(rate=350.0, pwf=2000.0, reservoir_pressure=2500.0)  # 1067.1 STB/d
ipr.composite_ipr(
    pwf=1000.0, reservoir_pressure=4000.0, bubble_point=2000.0, productivity_index=1.4
)
ipr.gilbert_choke(rate=1000.0, glr=1000.0, choke_size=32.0)  # 621 psi
```

**Decline curves**

```python
from wellcast.dca import DeclineModel

model = DeclineModel("auto").fit(rates).bootstrap(300)
model.forecast(365 * 5, econ_limit=20.0)
model.plot(rates, econ_limit=20.0).show()
```

**Features and machine learning**

```python
from wellcast.features import build_features
from wellcast.ml import Predictor

data = build_features(production)
vfm = Predictor("bhp").fit(data)
vfm.cross_validate(data, "lowo")  # leave one well out
vfm.predict_interval(data).head()  # 80 percent interval
vfm.explain(data).show()  # SHAP importance
```

**Diagnostics**

```python
from wellcast import diagnostics

injector = production.query("well == 'I-01'")
hall = diagnostics.hall_plot_data(injector, reservoir_pressure=3200.0)
diagnostics.hall_slope_change(hall)  # injectivity loss date and slope ratio
diagnostics.classify_water_mechanism(diagnostics.chan_plot_data(production))
diagnostics.detect_anomalies(production)
```

**Plotting**

```python
from wellcast import plotting

plotting.production_history(production, "P-01").show()
```

## Interactive app

<!-- Screenshot placeholder: add assets/app-screenshot.png and reference it here. -->

```bash
pip install "wellcast[all]"
wellcast app
```

Six pages (Data, Explore, Predict, Decline, Diagnose, Report) cover the whole
workflow, and **Load demo data** works with zero setup.

Live demo on Streamlit Community Cloud: **[wellcast.streamlit.app](https://wellcast.streamlit.app/)**

## Tutorials

| Notebook | Open |
|---|---|
| Quickstart | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olaflaitinen/wellcast/blob/main/notebooks/01_quickstart.ipynb) |
| Decline curve analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olaflaitinen/wellcast/blob/main/notebooks/02_decline_curve_analysis.ipynb) |
| Virtual flow metering | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olaflaitinen/wellcast/blob/main/notebooks/03_virtual_flow_metering.ipynb) |

Runnable scripts live in [`examples/`](https://github.com/olaflaitinen/wellcast/tree/main/examples).

## Documentation

The full documentation, with the engineering background and equations of every
module, tutorials and the API reference, is at
**[olaflaitinen.github.io/wellcast](https://olaflaitinen.github.io/wellcast)**.

## Roadmap

- **0.2.0**: vertical lift performance and nodal analysis, more PVT
  correlations.
- **0.3.0**: rate transient analysis and type curves.
- **1.0.0**: stable public API.

See [ROADMAP.md](https://github.com/olaflaitinen/wellcast/blob/main/ROADMAP.md).

## Contributing

Contributions are welcome. Please read
[CONTRIBUTING.md](https://github.com/olaflaitinen/wellcast/blob/main/CONTRIBUTING.md)
and the [Code of Conduct](https://github.com/olaflaitinen/wellcast/blob/main/CODE_OF_CONDUCT.md).
Questions go to [GitHub Discussions](https://github.com/olaflaitinen/wellcast/discussions)
and bugs to [GitHub Issues](https://github.com/olaflaitinen/wellcast/issues).

## Author

**Olaf Yunus Laitinen Imanov**<br>
Department of Mathematics and Statistics, University of Helsinki<br>
ORCID iD: [0009-0006-5184-0810](https://orcid.org/0009-0006-5184-0810)<br>
Email: [yunus.z.imanov@helsinki.fi](mailto:yunus.z.imanov@helsinki.fi)

## Citation

If wellcast contributes to your work, please cite it. The entry below is
generated from [CITATION.cff](https://github.com/olaflaitinen/wellcast/blob/main/CITATION.cff):

```bibtex
@misc{laitinenimanov2026wellcast,
  author = {Laitinen Imanov, Olaf Yunus},
  title = {wellcast: oil and gas well performance analysis for Python},
  version = {0.1.0},
  year = {2026},
  url = {https://olaflaitinen.github.io/wellcast},
  howpublished = {\url{https://github.com/olaflaitinen/wellcast}}
}
```

## License

wellcast is released under the [Mozilla Public License 2.0](https://github.com/olaflaitinen/wellcast/blob/main/LICENSE).
The MPL is a file-level copyleft: you may use wellcast in projects under any
license, including proprietary ones, but modifications to wellcast's own files
must be shared under the MPL.

## Acknowledgements

wellcast builds on NumPy, pandas, SciPy, scikit-learn, LightGBM, SHAP, plotly,
Streamlit and lasio, and on the published work of the engineers whose
correlations it implements, among them Arps, Standing, Vasquez and Beggs,
Beggs and Robinson, Hall and Yarborough, Vogel, Fetkovich, Gilbert, Chan, Hall,
and Havlena and Odeh.
