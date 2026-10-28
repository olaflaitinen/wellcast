# Machine learning

A virtual flow meter estimates a quantity that is expensive or unreliable to
measure from measurements that are always available. Typical uses: bottomhole
pressure after a gauge failure, or well rates between well tests.

```python
from wellcast.features import build_features
from wellcast.ml import Predictor

data = build_features(production)
model = Predictor("bhp", estimator="lightgbm").fit(data)
```

## Models

`Predictor` wraps a scikit-learn pipeline. Estimators: `lightgbm` (default),
`hist_gradient_boosting`, `random_forest` and `ridge`. Features default to all
numeric columns except the target, the well column, empty columns and
same-row features derived from the target.

## Validation

Random K-fold cross-validation overstates the accuracy of time-series models.
`cross_validate` offers two honest alternatives:

* `"timeseries"`: expanding-window splits on calendar dates. No model is
  trained on a date later than its test period, for any well. This measures
  extrapolation in time.
* `"lowo"`: leave one well out. This measures transfer to a well the model has
  never seen.

Metrics are reported per well and overall: MAE, RMSE, MAPE (ignoring zero
targets such as shut-in rates) and $R^2$.

## Prediction intervals

Two extra quantile-loss models (by default the 0.1 and 0.9 quantiles) give an
80 percent interval through `predict_interval`. Bounds are sorted per row,
because independently fitted quantile models can cross. Check the coverage on
held-out data before relying on it.

## Explanation

`shap_values` returns SHAP contributions (exact TreeSHAP for tree models,
linear SHAP for ridge, Lundberg and Lee, 2017), and `explain` draws the mean
absolute contribution per feature.

## Persistence

`save` and `load` use joblib and record the wellcast version. Joblib files are
pickles: only load files from a trusted source.
