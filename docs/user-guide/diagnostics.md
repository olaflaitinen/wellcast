# Diagnostics

## Chan plot

Chan (1995) plots the water-oil ratio $\mathrm{WOR} = q_w/q_o$ and its time
derivative $\mathrm{WOR}'$ against time on log-log axes. After breakthrough,
the derivative separates the mechanisms of excess water:

* **channeling** through a high-permeability layer: $\mathrm{WOR}'$ keeps
  rising;
* **coning** of bottom water: $\mathrm{WOR}'$ falls once the cone stabilizes.

`chan_plot_data` aggregates daily WOR into log-spaced time bins before
differentiating, and `classify_water_mechanism` screens wells by the log-log
slope of $\mathrm{WOR}'$. Treat the result as a screening hint and confirm it
with completion and reservoir data.

## Hall plot

For an injector, Hall (1963) plots

$$
\int_0^t (p_{wf} - \bar{p}_r)\,dt \quad \text{against} \quad W_i = \int_0^t i\,dt.
$$

For steady radial flow the slope is the inverse injectivity index. A rising
slope indicates plugging or growing skin; a falling slope indicates
stimulation or fracturing. `hall_plot_data` computes the curve and
`hall_slope_change` finds the strongest break with a two-segment fit. On the
synthetic injector it finds the injectivity loss within a day of the true date
and the slope ratio of $1/0.6$.

## Havlena-Odeh material balance

The general material balance in Havlena and Odeh's (1963) straight-line form
is

$$
F = N\,(E_o + m E_g + E_{fw}) + W_e
$$

with underground withdrawal
$F = N_p[B_o + (R_p - R_s)B_g] + W_p B_w - W_{inj} B_w$ and expansion terms
$E_o = (B_o - B_{oi}) + (R_{si} - R_s)B_g$, $E_g = B_{oi}(B_g/B_{gi} - 1)$ and
$E_{fw} = (1+m) B_{oi} \frac{c_w S_{wi} + c_f}{1 - S_{wi}} \Delta p$.

`havlena_odeh` tabulates these terms and `fit_ooip` returns $N$ as the slope
through the origin of $F$ against $E_t$. With aquifer influx,
$F/E_t = N + W_e/E_t$ lies above $N$ and varies in time.

## Anomaly detection

`detect_anomalies` combines explainable rules with an Isolation Forest
(Liu, Ting and Zhou, 2008):

| Detector | Finds |
|---|---|
| `impossible_<col>` | values outside physical limits |
| `dropout_<col>` | a pressure of exactly zero while the well flows |
| `flatline_<col>` | at least 5 identical non-zero pressure readings while flowing |
| `spike_<col>` | rate outliers at least 50 percent above the local median |
| `isolation_forest` | multivariate inconsistencies between signals and their changes |

Flagged rows are merged into periods. On the synthetic field the rules find
every injected spike, flatline and dropout while flagging about 0.05 percent
of normal days.

!!! warning "Slow drift"
    A gauge drifting by a few psi per day hides in the daily noise and is not
    reliably found by either detector. Monitor it through the residual between
    the measurement and a [virtual flow meter](ml.md) trained on a trusted
    period.
