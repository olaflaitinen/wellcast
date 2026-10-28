# Features

`wellcast.features` turns production data into model inputs.

## Physics features

`add_physics_features` adds, when the source columns exist:

| Feature | Definition |
|---|---|
| `liquid_rate` | $q_o + q_w$ |
| `gor` | $1000\,q_g / q_o$ (scf/STB, gas in Mscf/d) |
| `water_cut` | $q_w / (q_o + q_w)$ |
| `glr` | $1000\,q_g / q_L$ |
| `dp_tubing` | $p_{wf} - p_{wh}$ |
| `static_pressure`, `drawdown` | $\bar{p}_r$ and $\bar{p}_r - p_{wf}$ |
| `choke_normalized_rate` | $q_L$ per square inch of bean area |
| `cum_oil`, `cum_gas`, `cum_water` | cumulative volumes |
| `days_on_production` | cumulative producing time |

Ratios are NaN (not infinite) when the denominator is zero. When no reservoir
pressure is given, the static pressure is the last bottomhole pressure of the
most recent build-up, carried forward in time only.

## Leakage-safe history features

`add_lag_features` and `add_rolling_features` summarize the past of each well.
The rolling window at time $t$ covers rows $t - w$ to $t - 1$, so a value never
enters its own feature, and wells never mix, even when stored interleaved.

## Avoiding target leakage

Same-row physics features can contain the target: `drawdown` and `dp_tubing`
are computed from `bhp`. `features_using("bhp")` lists them, and the
[`Predictor`](ml.md) excludes them automatically.
