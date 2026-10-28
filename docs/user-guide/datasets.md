# Synthetic data

wellcast bundles no third-party data. Documentation, tests, tutorials and the
app demo all use `make_synthetic_field`, which simulates daily surface data
for a small oil field with known ground truth.

```python
from wellcast.datasets import make_synthetic_field

field = make_synthetic_field(n_wells=5, years=5, freq="D", seed=42)
field.production  # measurements in the canonical schema
field.labels  # shut_in, anomaly, anomaly_type, water_breakthrough per row
field.wells  # true parameters per well
```

## Physical model

Producers:

* liquid potential follows an Arps hyperbolic decline,
  $q = q_i (1 + b D_i t)^{-1/b}$, scaled by the choke opening as
  $(S/32)^{0.6}$;
* bottomhole pressure follows a linear IPR, $p_{wf} = \bar{p}_r - q_L/J$, and
  reservoir pressure falls in proportion to cumulative withdrawal;
* wellhead pressure follows the Gilbert (1954) choke equation,
  $p_{wh} = 435 \, R^{0.546} q_L / S^{1.89}$;
* the producing GOR equals the solution GOR above the bubble point and rises
  below it;
* water cut grows towards a plateau after breakthrough;
* shut-ins show zero rates and an exponential pressure build-up.

Each producer receives four labelled anomalies: a rate spike, a frozen
bottomhole gauge, a slow gauge drift and a wellhead pressure dropout.

Injectors inject at a constant surface pressure and lose 40 percent of their
injectivity at a known date, which shows on a [Hall plot](diagnostics.md).

The model is deliberately simple. It is meant for testing methods against
known answers, not for reservoir forecasting.
