# Quickstart

This page walks through the main workflow on the built-in synthetic field.
The same steps work on your own data: replace `make_synthetic_field` with
[`read_production`](user-guide/io.md).

## Load data

```python
import wellcast

field = wellcast.make_synthetic_field(n_wells=5, years=5, seed=42)
production = field.production  # canonical schema, indexed by date
production.head()
```

For a CSV or Excel export from a production database:

```python
production = wellcast.read_production(
    "field_export.csv",
    units={"oil_rate": "sm3/d", "whp": "bar"},  # converted to bbl/d and psi
)
```

## Check data quality

```python
report = wellcast.qc_report(production)
report[report["impossible"] > 0]
```

## Fit a decline curve

```python
rates = production.query("well == 'P-01'")["oil_rate"]
model = wellcast.DeclineModel().fit(rates).bootstrap(200)
model.summary(econ_limit=20.0)
model.eur_quantiles(econ_limit=20.0)  # P90, P50, P10
```

For every well at once:

```python
table = wellcast.fit_many(production, econ_limit=20.0)
```

## Diagnose

```python
from wellcast import diagnostics, plotting

anomalies = diagnostics.detect_anomalies(production)
plotting.anomaly_overlay(production, anomalies, "bhp", "P-01").show()
```

## Predict bottomhole pressure

```python
from wellcast.features import build_features
from wellcast.ml import Predictor

data = build_features(production)
model = Predictor("bhp").fit(data)
model.cross_validate(data, "lowo")  # leave one well out
```

## From the command line

```bash
wellcast demo --output field.csv
wellcast qc field.csv
wellcast dca field.csv --well P-01 --econ-limit 20 --bootstrap 200
wellcast app
```
