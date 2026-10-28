# Decline curve analysis

## Arps models

Arps (1945) described boundary-dominated rate decline with

$$
q(t) = \frac{q_i}{(1 + b D_i t)^{1/b}}
$$

where $q_i$ is the initial rate, $D_i$ the initial nominal decline and $b$ the
decline exponent. The limits are exponential decline ($b = 0$,
$q = q_i e^{-D_i t}$) and harmonic decline ($b = 1$). Cumulative production is

$$
N_p(t) = \frac{q_i}{(1 - b) D_i}\left[1 - (1 + b D_i t)^{(b-1)/b}\right]
$$

with closed forms for the two limits. wellcast evaluates these with `expm1`
and `log1p` to keep full precision for small $D_i t$ and for $b$ near 1.

In wellcast, time is in days and $D_i$ is the **nominal decline per year**.
`summary()` also reports the effective annual decline
$d_e = 1 - q(365.25)/q_i$.

## Fitting

```python
from wellcast.dca import DeclineModel

model = DeclineModel("auto").fit(rates)  # rates: Series with DatetimeIndex
model.summary(econ_limit=20.0)
```

* Zero and missing rates (shut-ins, gaps) are dropped.
* By default the fit starts at the peak of the 7-sample rolling median, so the
  ramp-up period is ignored (`from_peak=True`).
* Residuals are taken on $\ln q$, because rate errors are multiplicative, and
  a soft-L1 loss limits the influence of allocation spikes.
* Parameters are bounded ($0.01 \le b \le 2$ by default; $b > 1$ occurs during
  transient flow in tight reservoirs).
* With `model="auto"` the three models are fitted and the one with the
  smallest Akaike information criterion is kept,
  $\mathrm{AIC} = n \ln(\mathrm{RSS}/n) + 2k$.

## Forecast and EUR

```python
model.predict(dates_or_days)
model.cumulative(days)
model.eur(econ_limit=20.0, max_years=30)
model.remaining(econ_limit=20.0)
model.forecast(365 * 5, econ_limit=20.0)
```

EUR is the cumulative production from the start of decline to the time the
rate reaches the economic limit.

## Uncertainty

`bootstrap(n)` resamples the log-rate residuals, refits the selected model
`n` times and stores the parameter sets. `predict_interval` and
`eur_quantiles` then report **P90, P50 and P10** following the reserves
convention (SPE-PRMS): P90 is the low estimate, exceeded with 90 percent
probability, and P10 the high estimate.

## Many wells

`fit_many(df, econ_limit=...)` fits every well and returns one row per well.
Wells that cannot be fitted are reported in the `status` column instead of
stopping the batch. `DeclineModel.from_params` rebuilds a model from a row.

## Caveats

Arps models assume a constant operating regime. Choke changes, workovers,
artificial lift changes and water breakthrough all violate it. Fit stable
periods and inspect the plot before trusting an EUR.
