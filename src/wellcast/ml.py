# =============================================================================
# wellcast: machine learning predictors for virtual flow metering
# -----------------------------------------------------------------------------
# Module:      wellcast.ml
# Repository:  https://github.com/olaflaitinen/wellcast
# Author:      Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Copyright:   Copyright (c) 2026 Olaf Yunus Laitinen Imanov
# License:     Mozilla Public License 2.0
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# =============================================================================

"""Data-driven prediction of well rates and pressures.

A virtual flow meter (VFM) estimates a quantity that is expensive or
unreliable to measure, such as bottomhole pressure after a gauge failure or
the rate of a well between well tests, from measurements that are always
available (wellhead pressure and temperature, choke opening, and so on).

:class:`Predictor` wraps a scikit-learn :class:`~sklearn.pipeline.Pipeline`
(LightGBM by default), validates it with time-ordered or leave-one-well-out
splits, reports metrics per well, provides quantile prediction intervals and
explains predictions with SHAP.

Requires the ``ml`` extra: ``pip install 'wellcast[ml]'``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from wellcast.__about__ import __version__
from wellcast._validation import require_columns, require_extra
from wellcast.exceptions import DataValidationError, FitError
from wellcast.features import features_using

__all__ = [
    "Predictor",
    "metrics_by_well",
    "regression_metrics",
]

EstimatorName = Literal["lightgbm", "hist_gradient_boosting", "random_forest", "ridge"]
_TREE_ESTIMATORS = ("lightgbm", "hist_gradient_boosting", "random_forest")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def regression_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike
) -> dict[str, float]:
    r"""Mean absolute, root-mean-square and percentage errors and R2.

    Parameters
    ----------
    y_true, y_pred : array_like
        Observed and predicted values. Pairs with a missing value are ignored.

    Returns
    -------
    dict of str to float
        ``n``, ``mae``, ``rmse``, ``mape`` (percent) and ``r2``. MAPE ignores
        observations equal to zero (for example shut-in rates); R2 is NaN for
        fewer than two points or a constant target.

    Notes
    -----
    .. math::

        \mathrm{MAPE} = \frac{100}{n} \sum \frac{|y_i - \hat y_i|}{|y_i|},
        \quad R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}

    Examples
    --------
    >>> m = regression_metrics([100.0, 200.0], [110.0, 190.0])
    >>> m["mae"], m["rmse"], m["mape"]
    (10.0, 10.0, 7.5)
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise DataValidationError("y_true and y_pred must have the same length")
    keep = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[keep], yp[keep]
    n = int(yt.size)
    if n == 0:
        nan = float("nan")
        return {"n": 0.0, "mae": nan, "rmse": nan, "mape": nan, "r2": nan}
    err = yt - yp
    nonzero = np.abs(yt) > 0
    mape = (
        float(100.0 * np.mean(np.abs(err[nonzero]) / np.abs(yt[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / ss_tot if n > 1 and ss_tot > 0 else float("nan")
    return {
        "n": float(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": mape,
        "r2": r2,
    }


def metrics_by_well(
    y_true: pd.Series, y_pred: pd.Series, wells: pd.Series | None
) -> pd.DataFrame:
    """Compute :func:`regression_metrics` for each well and for all wells.

    Parameters
    ----------
    y_true, y_pred : pandas.Series
        Observed and predicted values, aligned by position.
    wells : pandas.Series or None
        Well names aligned with the values; None treats the data as one well.

    Returns
    -------
    pandas.DataFrame
        One row per well plus a row ``"all"``, columns ``n``, ``mae``,
        ``rmse``, ``mape`` and ``r2``.

    Examples
    --------
    >>> import pandas as pd
    >>> t = pd.Series([1.0, 2.0, 3.0, 4.0])
    >>> p = t + 1.0
    >>> metrics_by_well(t, p, pd.Series(["A", "A", "B", "B"]))["mae"].tolist()
    [1.0, 1.0, 1.0]
    """
    yt = y_true.to_numpy(dtype=np.float64, na_value=np.nan)
    yp = y_pred.to_numpy(dtype=np.float64, na_value=np.nan)
    rows: dict[str, dict[str, float]] = {}
    if wells is not None:
        names = wells.to_numpy()
        for well in sorted(pd.unique(names)):
            mask = names == well
            rows[str(well)] = regression_metrics(yt[mask], yp[mask])
    rows["all"] = regression_metrics(yt, yp)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["n"] = table["n"].astype(int)
    table.index.name = "well"
    result: pd.DataFrame = table
    return result


# -----------------------------------------------------------------------------
# Estimator factory
# -----------------------------------------------------------------------------


def _make_pipeline(
    estimator: str, random_state: int, params: dict[str, Any], quantile: float | None
) -> Any:
    pipeline_mod = require_extra("sklearn.pipeline", "ml")
    impute_mod = require_extra("sklearn.impute", "ml")
    imputer = impute_mod.SimpleImputer(strategy="median", keep_empty_features=True)

    if estimator == "lightgbm":
        lgb = require_extra("lightgbm", "ml")
        options: dict[str, Any] = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "random_state": random_state,
            "verbose": -1,
        }
        if quantile is not None:
            options.update(objective="quantile", alpha=quantile)
        options.update(params)
        # LightGBM handles missing values natively; no imputation needed.
        return pipeline_mod.Pipeline([("model", lgb.LGBMRegressor(**options))])

    ensemble = require_extra("sklearn.ensemble", "ml")
    if quantile is not None:
        # Quantile intervals for non-LightGBM point models use histogram
        # gradient boosting with the pinball loss.
        return pipeline_mod.Pipeline(
            [
                (
                    "model",
                    ensemble.HistGradientBoostingRegressor(
                        loss="quantile", quantile=quantile, random_state=random_state
                    ),
                )
            ]
        )
    if estimator == "hist_gradient_boosting":
        return pipeline_mod.Pipeline(
            [
                (
                    "model",
                    ensemble.HistGradientBoostingRegressor(
                        random_state=random_state, **params
                    ),
                )
            ]
        )
    if estimator == "random_forest":
        options = {
            "n_estimators": 200,
            "min_samples_leaf": 2,
            "random_state": random_state,
        }
        options.update(params)
        return pipeline_mod.Pipeline(
            [("impute", imputer), ("model", ensemble.RandomForestRegressor(**options))]
        )
    if estimator == "ridge":
        linear = require_extra("sklearn.linear_model", "ml")
        preprocessing = require_extra("sklearn.preprocessing", "ml")
        return pipeline_mod.Pipeline(
            [
                ("impute", imputer),
                ("scale", preprocessing.StandardScaler()),
                ("model", linear.Ridge(**params)),
            ]
        )
    raise DataValidationError(f"unknown estimator {estimator!r}")


# -----------------------------------------------------------------------------
# Predictor
# -----------------------------------------------------------------------------


class Predictor:
    """Supervised model for a well rate or pressure (virtual flow metering).

    Parameters
    ----------
    target : str, optional
        Column to predict, for example ``"bhp"`` or ``"oil_rate"``.
        Default ``"bhp"``.
    features : sequence of str, optional
        Feature columns. By default every numeric column except the target,
        the well column and the same-row features derived from the target
        (see :func:`wellcast.features.features_using`), which would leak it.
    estimator : {"lightgbm", "hist_gradient_boosting", "random_forest", "ridge"}
        Point model. Default ``"lightgbm"``.
    quantiles : tuple of float or None, optional
        Lower and upper quantiles of the prediction interval, fitted as
        separate quantile-loss models. Default (0.1, 0.9), an 80 percent
        interval. None disables intervals.
    exclude_shut_ins : bool, optional
        Drop rows with zero producing hours from training. Default True.
    by : str, optional
        Well column. Default ``"well"``.
    random_state : int, optional
        Seed for the estimators. Default 0.
    params : dict, optional
        Extra keyword arguments for the point estimator.

    Attributes
    ----------
    features_ : list of str
        Feature columns used in the fit.
    pipeline_ : sklearn.pipeline.Pipeline
        Fitted point model.
    quantile_pipelines_ : dict of float to Pipeline
        Fitted quantile models.

    Examples
    --------
    >>> from wellcast.datasets import make_synthetic_field
    >>> from wellcast.features import build_features
    >>> field = make_synthetic_field(n_wells=2, years=2, n_injectors=0)
    >>> data = build_features(field.production)
    >>> model = Predictor(target="bhp", quantiles=None).fit(data)
    >>> "bhp" in model.features_, "bhp_lag1" in model.features_
    (False, True)
    >>> bool(model.score(data).loc["all", "mape"] < 5.0)
    True
    """

    def __init__(
        self,
        target: str = "bhp",
        features: Sequence[str] | None = None,
        *,
        estimator: EstimatorName = "lightgbm",
        quantiles: tuple[float, float] | None = (0.1, 0.9),
        exclude_shut_ins: bool = True,
        by: str = "well",
        random_state: int = 0,
        params: dict[str, Any] | None = None,
    ) -> None:
        if estimator not in (*_TREE_ESTIMATORS, "ridge"):
            raise DataValidationError(f"unknown estimator {estimator!r}")
        if quantiles is not None and not 0.0 < quantiles[0] < quantiles[1] < 1.0:
            raise DataValidationError("quantiles must satisfy 0 < low < high < 1")
        self.target = target
        self.features = None if features is None else list(features)
        self.estimator = estimator
        self.quantiles = quantiles
        self.exclude_shut_ins = exclude_shut_ins
        self.by = by
        self.random_state = random_state
        self.params = dict(params or {})
        self.features_: list[str] = []
        self.pipeline_: Any = None
        self.quantile_pipelines_: dict[float, Any] = {}

    # -- data preparation -----------------------------------------------------

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        if self.features is not None:
            require_columns(df, self.features)
            if self.target in self.features:
                raise DataValidationError("the target cannot be a feature")
            return list(self.features)
        leaky = set(features_using(self.target)) | {self.target, self.by}
        numeric = df.select_dtypes(include=["number", "bool"]).columns
        # Columns with no observed value (for example injection rates in a
        # producer-only data set) carry no information and break some
        # estimators' binning, so they are skipped.
        chosen = [str(c) for c in numeric if c not in leaky and df[c].notna().any()]
        if not chosen:
            raise DataValidationError("no usable feature columns")
        return chosen

    def _training_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        require_columns(df, [self.target])
        rows: pd.DataFrame = df[df[self.target].notna()]
        if self.exclude_shut_ins and "on_stream_hours" in rows.columns:
            rows = rows[rows["on_stream_hours"].fillna(24.0) > 0]
        if len(rows) < 10:
            raise FitError("at least 10 training rows with a known target are needed")
        return rows

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        require_columns(df, self.features_)
        matrix: pd.DataFrame = df[self.features_].astype("float64")
        return matrix

    # -- fitting and prediction -----------------------------------------------

    def fit(self, df: pd.DataFrame) -> Predictor:
        """Fit the point model and the quantile models.

        Parameters
        ----------
        df : pandas.DataFrame
            Training data containing the target and the features.

        Returns
        -------
        Predictor
            ``self``.

        Raises
        ------
        FitError
            If fewer than 10 usable rows remain.
        DataValidationError
            If columns are missing.
        MissingDependencyError
            If the ``ml`` extra is not installed.
        """
        rows = self._training_rows(df)
        self.features_ = self._resolve_features(rows)
        x = self._matrix(rows)
        y = rows[self.target].to_numpy(dtype=np.float64)
        self.pipeline_ = _make_pipeline(
            self.estimator, self.random_state, self.params, None
        ).fit(x, y)
        self.quantile_pipelines_ = {}
        for q in self.quantiles or ():
            self.quantile_pipelines_[q] = _make_pipeline(
                self.estimator, self.random_state, {}, q
            ).fit(x, y)
        return self

    def _check_fitted(self) -> None:
        if self.pipeline_ is None:
            raise FitError("predictor is not fitted; call fit() first")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict the target.

        Parameters
        ----------
        df : pandas.DataFrame
            Rows with the feature columns.

        Returns
        -------
        pandas.Series
            Predictions aligned with ``df``, named ``<target>_pred``.

        Raises
        ------
        FitError
            If the predictor is not fitted.
        """
        self._check_fitted()
        values = self.pipeline_.predict(self._matrix(df))
        prediction: pd.Series = pd.Series(
            values, index=df.index, name=f"{self.target}_pred"
        )
        return prediction

    def predict_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict with a quantile interval.

        Parameters
        ----------
        df : pandas.DataFrame
            Rows with the feature columns.

        Returns
        -------
        pandas.DataFrame
            Columns ``lower``, ``prediction`` and ``upper``. The bounds are
            sorted per row, because independently fitted quantile models can
            cross.

        Raises
        ------
        FitError
            If the predictor is not fitted or was created with
            ``quantiles=None``.
        """
        self._check_fitted()
        if not self.quantile_pipelines_:
            raise FitError("no quantile models; create the Predictor with quantiles")
        x = self._matrix(df)
        low_q, high_q = sorted(self.quantile_pipelines_)
        low = self.quantile_pipelines_[low_q].predict(x)
        high = self.quantile_pipelines_[high_q].predict(x)
        bounds = np.sort(np.vstack([low, high]), axis=0)
        interval: pd.DataFrame = pd.DataFrame(
            {
                "lower": bounds[0],
                "prediction": self.pipeline_.predict(x),
                "upper": bounds[1],
            },
            index=df.index,
        )
        return interval

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-well metrics of the fitted model on ``df``.

        Parameters
        ----------
        df : pandas.DataFrame
            Rows with the target and the features.

        Returns
        -------
        pandas.DataFrame
            See :func:`metrics_by_well`.

        Raises
        ------
        FitError
            If the predictor is not fitted.
        """
        rows = self._training_rows(df)
        wells = rows[self.by] if self.by in rows.columns else None
        return metrics_by_well(rows[self.target], self.predict(rows), wells)

    # -- validation -------------------------------------------------------------

    def _splits(
        self, rows: pd.DataFrame, method: str, n_splits: int
    ) -> Iterator[tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]]:
        if method == "timeseries":
            model_selection = require_extra("sklearn.model_selection", "ml")
            dates = np.sort(rows.index.unique())
            if len(dates) <= n_splits:
                raise DataValidationError("not enough distinct dates for the splits")
            splitter = model_selection.TimeSeriesSplit(n_splits=n_splits)
            row_dates = rows.index.to_numpy()
            for train_idx, test_idx in splitter.split(dates):
                # Split on calendar dates, not rows, so that no well is
                # trained on a date later than the test period of another.
                train = np.isin(row_dates, dates[train_idx])
                test = np.isin(row_dates, dates[test_idx])
                yield train, test
        elif method == "lowo":
            if self.by not in rows.columns:
                raise DataValidationError(
                    f"leave-one-well-out needs a {self.by!r} column"
                )
            wells = rows[self.by].to_numpy()
            names = sorted(pd.unique(wells))
            if len(names) < 2:
                raise DataValidationError("leave-one-well-out needs at least two wells")
            for name in names:
                test = wells == name
                yield ~test, test
        else:
            raise DataValidationError(f"unknown validation method {method!r}")

    def cross_validate(
        self,
        df: pd.DataFrame,
        method: Literal["timeseries", "lowo"] = "timeseries",
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Out-of-sample validation with per-well metrics.

        Parameters
        ----------
        df : pandas.DataFrame
            Data with the target and the features (a ``DatetimeIndex`` for
            ``"timeseries"``).
        method : {"timeseries", "lowo"}, optional
            ``"timeseries"``: expanding-window splits on calendar dates
            (:class:`sklearn.model_selection.TimeSeriesSplit`), testing how the
            model extrapolates in time. ``"lowo"``: leave one well out, testing
            how it transfers to an unseen well. Default ``"timeseries"``.
        n_splits : int, optional
            Number of time splits. Default 5.

        Returns
        -------
        pandas.DataFrame
            Columns ``fold``, ``well``, ``n``, ``mae``, ``rmse``, ``mape`` and
            ``r2``, one row per (fold, well) plus an ``"all"`` row per fold.

        Raises
        ------
        DataValidationError
            If the method is unknown or the data cannot be split.
        """
        rows = self._training_rows(df)
        parts = []
        for fold, (train, test) in enumerate(self._splits(rows, method, n_splits)):
            clone = Predictor(
                self.target,
                self.features,
                estimator=self.estimator,
                quantiles=None,
                exclude_shut_ins=self.exclude_shut_ins,
                by=self.by,
                random_state=self.random_state,
                params=self.params,
            ).fit(rows[train])
            test_rows = rows[test]
            wells = test_rows[self.by] if self.by in test_rows.columns else None
            table = metrics_by_well(
                test_rows[self.target], clone.predict(test_rows), wells
            )
            parts.append(table.reset_index().assign(fold=fold))
        result = pd.concat(parts, ignore_index=True)
        ordered: pd.DataFrame = result[
            ["fold", "well", "n", "mae", "rmse", "mape", "r2"]
        ]
        return ordered

    # -- explanation ------------------------------------------------------------

    def shap_values(self, df: pd.DataFrame, *, max_samples: int = 500) -> pd.DataFrame:
        """SHAP contributions of each feature to each prediction.

        Parameters
        ----------
        df : pandas.DataFrame
            Rows to explain; a random subset of ``max_samples`` rows is used
            when larger.
        max_samples : int, optional
            Maximum rows to explain. Default 500.

        Returns
        -------
        pandas.DataFrame
            One column per feature, in target units.

        Raises
        ------
        FitError
            If the predictor is not fitted.
        MissingDependencyError
            If ``shap`` is not installed.

        Notes
        -----
        Tree models use the exact TreeSHAP algorithm, the ridge model the
        linear SHAP solution. Lundberg, S. M. and Lee, S.-I. (2017), "A unified
        approach to interpreting model predictions", NeurIPS.
        """
        self._check_fitted()
        with warnings.catch_warnings():
            # shap imports tqdm.auto, which warns inside notebooks that lack
            # ipywidgets; the notice is about progress bars only.
            warnings.filterwarnings("ignore", message="IProgress not found")
            shap = cast(Any, require_extra("shap", "ml"))
        x = self._matrix(df)
        if len(x) > max_samples:
            x = x.sample(max_samples, random_state=self.random_state)
        steps = self.pipeline_.steps
        transformed: Any = x
        for _, step in steps[:-1]:
            transformed = step.transform(transformed)
        model = steps[-1][1]
        with warnings.catch_warnings():
            # shap emits deprecation notices from its own dependencies; they
            # are not actionable for wellcast users.
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            if self.estimator in _TREE_ESTIMATORS:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, transformed)
            values = np.asarray(explainer.shap_values(transformed), dtype=np.float64)
        contributions: pd.DataFrame = pd.DataFrame(
            values, columns=self.features_, index=x.index
        )
        return contributions

    def explain(
        self, df: pd.DataFrame, *, max_samples: int = 500, top: int = 15
    ) -> Any:
        """Bar chart of mean absolute SHAP values (global feature importance).

        Parameters
        ----------
        df : pandas.DataFrame
            Rows to explain.
        max_samples : int, optional
            Maximum rows to explain. Default 500.
        top : int, optional
            Number of features shown. Default 15.

        Returns
        -------
        plotly.graph_objects.Figure
            Horizontal bar chart, most important feature on top.

        Raises
        ------
        MissingDependencyError
            If ``shap`` or ``plotly`` is not installed.
        """
        from wellcast.plotting import importance_bar

        importance = (
            self.shap_values(df, max_samples=max_samples)
            .abs()
            .mean()
            .sort_values(ascending=True)
            .tail(top)
        )
        return importance_bar(
            importance,
            title=f"Feature importance for {self.target} (mean |SHAP|)",
            x_title=f"mean |SHAP value| ({self.target} units)",
        )

    # -- persistence ------------------------------------------------------------

    def save(self, path: str | PathLike[str]) -> None:
        """Save the predictor with joblib.

        Parameters
        ----------
        path : str or path-like
            Output file, conventionally ``*.joblib``.

        Raises
        ------
        FitError
            If the predictor is not fitted.
        """
        self._check_fitted()
        joblib = cast(Any, require_extra("joblib", "ml"))
        joblib.dump({"wellcast_version": __version__, "predictor": self}, path)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Predictor:
        """Load a predictor saved with :meth:`save`.

        Joblib files are pickles: load only files from a trusted source, since
        unpickling can execute arbitrary code.

        Parameters
        ----------
        path : str or path-like
            File written by :meth:`save`.

        Returns
        -------
        Predictor
            The restored predictor.

        Raises
        ------
        DataValidationError
            If the file does not contain a wellcast predictor.
        """
        joblib = cast(Any, require_extra("joblib", "ml"))
        payload = joblib.load(path)
        if not (
            isinstance(payload, dict) and isinstance(payload.get("predictor"), cls)
        ):
            raise DataValidationError(f"{path} does not contain a wellcast Predictor")
        if payload.get("wellcast_version") != __version__:
            warnings.warn(
                f"predictor was saved with wellcast {payload.get('wellcast_version')}, "
                f"loading with {__version__}",
                UserWarning,
                stacklevel=2,
            )
        predictor: Predictor = payload["predictor"]
        return predictor

    def __repr__(self) -> str:
        state = "fitted" if self.pipeline_ is not None else "unfitted"
        return (
            f"Predictor(target={self.target!r}, estimator={self.estimator!r}, {state})"
        )


# =============================================================================
# End of module: wellcast.ml
# =============================================================================
