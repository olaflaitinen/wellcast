# Plotting

`wellcast.plotting` returns interactive plotly figures in one consistent
style, the `"wellcast"` template.

| Function | Chart |
|---|---|
| `production_history` | Rates and pressures of a well in two stacked panels |
| `decline_plot` | Observed rates, Arps fit, P90 to P10 band, economic limit |
| `ipr_plot` | One or more IPR curves with an optional test point |
| `predicted_vs_actual` | Cross-plot with a 1:1 line |
| `prediction_interval_plot` | Time series with a prediction interval |
| `chan_plot` | WOR and WOR' on log-log axes |
| `hall_plot` | Hall integral with the detected slope change |
| `anomaly_overlay` | A signal with flagged periods shaded and labelled |
| `importance_bar` | Feature importance bars |

## Design rules

* **Color follows the entity.** Fluids keep fixed colors everywhere (oil
  green, gas red, water blue) together with a dash pattern, so identity never
  relies on color alone. Other series use the categorical palette in a fixed
  order.
* **One axis per quantity.** Rates and pressures go into separate panels that
  share the time axis instead of a second y-axis.
* **At most three colored series in scatter charts.** With more wells, points
  share one color and the well name appears on hover.
* **Status color plus a label.** Anomaly bands are always labelled with the
  detector name.

The palette was checked for color-vision deficiency separation.
`register_theme()` makes the template available as `template="wellcast"` for
your own figures.
