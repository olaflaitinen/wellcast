# =============================================================================
# wellcast: Arps decline curve analysis with model selection and uncertainty
# -----------------------------------------------------------------------------
# Module:      wellcast.dca
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

r"""Arps decline curve analysis.

Time is measured in days from the start of the fitted decline, rates in
volume per day (for example bbl/d) and cumulative volumes in the same volume
unit. The nominal decline rate ``di`` is expressed per year, as is customary
in reserves reporting.

The three Arps (1945) models are

.. math::

    q(t) = \begin{cases}
    q_i e^{-D_i t} & b = 0 \ \text{(exponential)} \\
    q_i (1 + b D_i t)^{-1/b} & 0 < b < 1 \ \text{(hyperbolic)} \\
    q_i / (1 + D_i t) & b = 1 \ \text{(harmonic)}
    \end{cases}

Arps, J. J. (1945), "Analysis of decline curves", Transactions of the AIME
160(1), 228-247, SPE-945228-G.

Probabilistic bands follow the petroleum reserves convention (SPE-PRMS):
P90 is the low estimate (exceeded with 90 percent probability), P50 the best
estimate and P10 the high estimate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import curve_fit

from wellcast._validation import as_float_array, check_positive, require_extra
from wellcast.exceptions import DataValidationError, FitError

__all__ = [
    "DeclineModel",
    "ModelName",
    "arps_cumulative",
    "arps_rate",
    "arps_time_to_rate",
    "fit_many",
]

FloatArray = npt.NDArray[np.float64]
FloatOrArray = float | FloatArray
ModelName = Literal["exponential", "hyperbolic", "harmonic"]
_MODELS: tuple[ModelName, ...] = ("exponential", "harmonic", "hyperbolic")
_DAYS_PER_YEAR = 365.25
# b values this close to 0 or 1 use the exponential or harmonic closed form,
# avoiding the removable singularities of the hyperbolic expressions.
_B_EPS = 1e-6


def _out(value: FloatArray) -> FloatOrArray:
    return float(value) if value.ndim == 0 else value


# -----------------------------------------------------------------------------
# Arps equations
# -----------------------------------------------------------------------------


def arps_rate(t: npt.ArrayLike, qi: float, di: float, b: float) -> FloatOrArray:
    r"""Arps decline rate.

    Parameters
    ----------
    t : array_like
        Time since the start of decline, days.
    qi : float
        Initial rate at ``t = 0``, volume/day.
    di : float
        Initial nominal decline rate, 1/year.
    b : float
        Arps exponent: 0 exponential, 1 harmonic, otherwise hyperbolic.

    Returns
    -------
    float or numpy.ndarray
        Rate, volume/day.

    Notes
    -----
    See the module docstring for the equations; :math:`D_i` is converted to
    1/day as ``di / 365.25``.

    Examples
    --------
    >>> round(arps_rate(365.25, 1000.0, 0.5, 0.0), 2)
    606.53
    >>> round(arps_rate(365.25, 1000.0, 0.5, 1.0), 2)
    666.67
    """
    tt = as_float_array(t, "t")
    d = di / _DAYS_PER_YEAR
    if b < _B_EPS:
        return _out(qi * np.exp(-d * tt))
    return _out(qi * np.power(1.0 + b * d * tt, -1.0 / b))


def arps_cumulative(t: npt.ArrayLike, qi: float, di: float, b: float) -> FloatOrArray:
    r"""Cumulative production of an Arps decline from ``t = 0``.

    Parameters
    ----------
    t : array_like
        Time since the start of decline, days.
    qi : float
        Initial rate, volume/day.
    di : float
        Initial nominal decline rate, 1/year.
    b : float
        Arps exponent.

    Returns
    -------
    float or numpy.ndarray
        Cumulative volume.

    Notes
    -----
    With :math:`D = D_i / 365.25` (1/day):

    .. math::

        N_p(t) = \begin{cases}
        \frac{q_i}{D}\left(1 - e^{-Dt}\right) & b = 0 \\
        \frac{q_i}{(1-b) D}\left[1 - (1 + bDt)^{(b-1)/b}\right] & b \ne 0, 1 \\
        \frac{q_i}{D}\ln(1 + Dt) & b = 1
        \end{cases}

    Examples
    --------
    >>> round(arps_cumulative(365.25, 1000.0, 0.5, 0.0), 0)
    287429.0
    """
    tt = as_float_array(t, "t")
    d = di / _DAYS_PER_YEAR
    if b < _B_EPS:
        return _out(qi / d * (1.0 - np.exp(-d * tt)))
    if abs(b - 1.0) < _B_EPS:
        return _out(qi / d * np.log1p(d * tt))
    return _out(
        qi / ((1.0 - b) * d) * (1.0 - np.power(1.0 + b * d * tt, (b - 1.0) / b))
    )


def arps_time_to_rate(q: npt.ArrayLike, qi: float, di: float, b: float) -> FloatOrArray:
    r"""Time at which an Arps decline reaches rate ``q``.

    Parameters
    ----------
    q : array_like
        Target rate, volume/day.
    qi : float
        Initial rate, volume/day.
    di : float
        Initial nominal decline rate, 1/year.
    b : float
        Arps exponent.

    Returns
    -------
    float or numpy.ndarray
        Time in days (0 when ``q >= qi``).

    Notes
    -----
    :math:`t = \ln(q_i/q)/D` for :math:`b = 0`, otherwise
    :math:`t = [(q_i/q)^b - 1]/(bD)`.

    Examples
    --------
    >>> round(arps_time_to_rate(500.0, 1000.0, 0.5, 1.0), 2)
    730.5
    """
    qq = check_positive(q, "q")
    d = di / _DAYS_PER_YEAR
    ratio = np.maximum(qi / qq, 1.0)
    if b < _B_EPS:
        return _out(np.log(ratio) / d)
    return _out((np.power(ratio, b) - 1.0) / (b * d))


# -----------------------------------------------------------------------------
# Fitting helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fit:
    name: ModelName
    qi: float
    di: float
    b: float
    rss: float
    aic: float


def _log_model(name: ModelName) -> tuple[Callable[..., FloatArray], int]:
    # Model of log(q) and its number of free parameters.
    if name == "exponential":
        return (lambda t, qi, di: np.log(qi) - di / _DAYS_PER_YEAR * t), 2
    if name == "harmonic":
        return (lambda t, qi, di: np.log(qi) - np.log1p(di / _DAYS_PER_YEAR * t)), 2
    return (
        lambda t, qi, di, b: np.log(qi) - np.log1p(b * di / _DAYS_PER_YEAR * t) / b
    ), 3


def _fit_one(
    name: ModelName,
    t: FloatArray,
    log_q: FloatArray,
    b_bounds: tuple[float, float],
    loss: str,
    p0: Sequence[float] | None = None,
) -> _Fit:
    func, k = _log_model(name)
    q_max = float(np.exp(np.max(log_q)))
    if p0 is None:
        # Initial guess: first rate and the log-linear (exponential) slope.
        slope = np.polyfit(t, log_q, 1)[0] if np.ptp(t) > 0 else -1e-3
        di0 = float(np.clip(-slope * _DAYS_PER_YEAR, 1e-3, 10.0))
        p0 = [float(np.exp(log_q[0])), di0, 0.5][:k]
    lower = [1e-6 * q_max, 1e-6, b_bounds[0]][:k]
    upper = [10.0 * q_max, 50.0, b_bounds[1]][:k]
    start = np.clip(np.asarray(list(p0)[:k], dtype=np.float64), lower, upper)
    try:
        params, _ = curve_fit(
            func,
            t,
            log_q,
            p0=start,
            bounds=(lower, upper),
            method="trf",
            loss=loss,
            f_scale=0.1,
            max_nfev=5000,
        )
    except (RuntimeError, ValueError) as exc:
        raise FitError(f"{name} fit failed: {exc}") from exc
    residuals = log_q - func(t, *params)
    rss = float(np.sum(residuals**2))
    n = len(t)
    # AIC for Gaussian errors in log-rate: n ln(RSS/n) + 2k.
    aic = n * np.log(max(rss, 1e-300) / n) + 2 * k
    b = {"exponential": 0.0, "harmonic": 1.0}.get(name, float(params[-1]))
    return _Fit(name, float(params[0]), float(params[1]), b, rss, float(aic))


# -----------------------------------------------------------------------------
# DeclineModel
# -----------------------------------------------------------------------------


class DeclineModel:
    r"""Fit, forecast and evaluate an Arps decline.

    Parameters
    ----------
    model : {"auto", "exponential", "hyperbolic", "harmonic"}, optional
        Model to fit. ``"auto"`` fits all three and keeps the lowest AIC.
        Default ``"auto"``.
    b_bounds : tuple of float, optional
        Bounds on the hyperbolic exponent. Default (0.01, 2.0); values above
        1 occur during transient (unconventional) flow.
    from_peak : bool, optional
        Start the fit at the peak of the 7-sample rolling median rate, so the
        ramp-up period is ignored. Default True.
    loss : str, optional
        Loss passed to :func:`scipy.optimize.least_squares`. The default
        ``"soft_l1"`` (scale 0.1 in log-rate, about 10 percent) limits the
        influence of rate spikes; use ``"linear"`` for ordinary least squares.

    Attributes
    ----------
    model_ : str
        Selected model name.
    qi_, di_, b_ : float
        Fitted initial rate (volume/day), nominal decline (1/year) and
        exponent.
    aic_ : dict of str to float
        AIC of every candidate model.
    start_ : pandas.Timestamp or None
        Date of ``t = 0`` when fitted on a date-indexed Series.
    t_offset_ : float
        Time of the decline start (``t = 0``) on the input time axis, days.
        Non-zero when ``from_peak`` skipped a ramp-up period.
    t_last_ : float
        Time of the last observation used in the fit, days after decline
        start.
    n_obs_ : int
        Number of observations used.

    Notes
    -----
    Zero, negative and missing rates (shut-ins and gaps) are excluded before
    fitting. Fitting minimizes residuals of :math:`\ln q`, which treats rate
    errors as multiplicative.

    Examples
    --------
    >>> import numpy as np
    >>> from wellcast.dca import DeclineModel, arps_rate
    >>> t = np.arange(0.0, 1000.0, 10.0)
    >>> q = arps_rate(t, 1500.0, 0.8, 0.5)
    >>> m = DeclineModel("hyperbolic").fit(q, time=t)
    >>> round(m.qi_, 1), round(m.di_, 3), round(m.b_, 3)
    (1500.0, 0.8, 0.5)
    >>> round(m.eur(econ_limit=50.0) / 1e6, 4)
    1.1196
    """

    def __init__(
        self,
        model: Literal["auto", "exponential", "hyperbolic", "harmonic"] = "auto",
        *,
        b_bounds: tuple[float, float] = (0.01, 2.0),
        from_peak: bool = True,
        loss: str = "soft_l1",
    ) -> None:
        if model != "auto" and model not in _MODELS:
            raise DataValidationError(f"unknown model {model!r}")
        if not 0.0 < b_bounds[0] < b_bounds[1]:
            raise DataValidationError("b_bounds must satisfy 0 < low < high")
        self.model = model
        self.b_bounds = b_bounds
        self.from_peak = from_peak
        self.loss = loss
        self._fitted = False
        self._samples: FloatArray | None = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        qi: float,
        di: float,
        b: float,
        *,
        start: pd.Timestamp | str | None = None,
    ) -> DeclineModel:
        """Create a fitted model from known parameters.

        Parameters
        ----------
        qi : float
            Initial rate, volume/day.
        di : float
            Initial nominal decline, 1/year.
        b : float
            Arps exponent.
        start : pandas.Timestamp or str, optional
            Date of ``t = 0``, enabling date-based prediction.

        Returns
        -------
        DeclineModel
            A model that can predict and compute EUR but has no bootstrap
            samples.

        Examples
        --------
        >>> m = DeclineModel.from_params(1000.0, 0.5, 0.0)
        >>> m.model_
        'exponential'
        """
        check_positive(qi, "qi")
        check_positive(di, "di")
        if b < 0:
            raise DataValidationError("b must be non-negative")
        model = cls()
        name: ModelName
        if b < _B_EPS:
            name = "exponential"
        elif abs(b - 1.0) < _B_EPS:
            name = "harmonic"
        else:
            name = "hyperbolic"
        model._set(name, qi, di, b)
        model.start_ = None if start is None else pd.Timestamp(start)
        model.t_last_ = 0.0
        model.t_offset_ = 0.0
        model.n_obs_ = 0
        model.aic_ = {}
        model.residual_std_ = float("nan")
        return model

    def _set(self, name: ModelName, qi: float, di: float, b: float) -> None:
        self.model_: ModelName = name
        self.qi_ = qi
        self.di_ = di
        self.b_ = b
        self._fitted = True

    # -- fitting ----------------------------------------------------------------

    def _prepare(
        self, rates: pd.Series | npt.ArrayLike, time: npt.ArrayLike | None
    ) -> tuple[FloatArray, FloatArray, pd.Timestamp | None]:
        start: pd.Timestamp | None = None
        if isinstance(rates, pd.Series) and isinstance(rates.index, pd.DatetimeIndex):
            if time is not None:
                raise DataValidationError("pass either a date-indexed Series or time")
            dates = rates.index
            q = rates.to_numpy(dtype=np.float64, na_value=np.nan)
            t_abs = np.asarray(
                (dates - dates[0]) / pd.Timedelta(days=1), dtype=np.float64
            )
            start = pd.Timestamp(dates[0])
        else:
            q = as_float_array(rates, "rates").ravel()
            if time is None:
                t_abs = np.arange(len(q), dtype=np.float64)
            else:
                t_abs = as_float_array(time, "time").ravel()
                if t_abs.shape != q.shape:
                    raise DataValidationError("rates and time must have equal length")
        valid = np.isfinite(q) & np.isfinite(t_abs) & (q > 0)
        q, t_abs = q[valid], t_abs[valid]
        order = np.argsort(t_abs, kind="stable")
        q, t_abs = q[order], t_abs[order]
        if self.from_peak and len(q):
            smooth = pd.Series(q).rolling(7, center=True, min_periods=1).median()
            peak = int(np.argmax(smooth.to_numpy()))
            q, t_abs = q[peak:], t_abs[peak:]
        if len(q) < 4:
            raise FitError("at least 4 positive rate observations are needed")
        t0 = float(t_abs[0])
        self.t_offset_ = t0
        if start is not None:
            start = start + pd.Timedelta(days=t0)
        return t_abs - t0, q, start

    def fit(
        self,
        rates: pd.Series | npt.ArrayLike,
        time: npt.ArrayLike | None = None,
    ) -> DeclineModel:
        """Fit the decline model.

        Parameters
        ----------
        rates : pandas.Series or array_like
            Rates in volume/day. A Series with a ``DatetimeIndex`` defines the
            time axis itself; otherwise ``time`` (days) is used, or the sample
            number when ``time`` is omitted.
        time : array_like, optional
            Observation times in days, for array input.

        Returns
        -------
        DeclineModel
            The fitted model (``self``).

        Raises
        ------
        FitError
            If fewer than 4 positive observations remain or every candidate
            fit fails.
        DataValidationError
            If inputs are inconsistent.
        """
        t, q, start = self._prepare(rates, time)
        log_q = np.log(q)
        candidates = _MODELS if self.model == "auto" else (self.model,)
        fits: dict[str, _Fit] = {}
        errors: list[str] = []
        for name in candidates:
            try:
                fits[name] = _fit_one(name, t, log_q, self.b_bounds, self.loss)
            except FitError as exc:  # pragma: no cover - defensive
                errors.append(str(exc))
        if not fits:  # pragma: no cover - defensive
            raise FitError("; ".join(errors))
        best = min(fits.values(), key=lambda f: f.aic)
        self._set(best.name, best.qi, best.di, best.b)
        self.aic_ = {name: f.aic for name, f in fits.items()}
        self.start_ = start
        self.t_last_ = float(t[-1])
        self.n_obs_ = len(t)
        self._t = t
        self._log_q = log_q
        fitted = np.log(np.asarray(arps_rate(t, best.qi, best.di, best.b)))
        self._residuals = log_q - fitted
        self.residual_std_ = float(np.std(self._residuals, ddof=0))
        self._samples = None
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise FitError("model is not fitted; call fit() first")

    # -- evaluation -------------------------------------------------------------

    def _to_days(self, t: npt.ArrayLike | pd.DatetimeIndex) -> FloatArray:
        if isinstance(t, pd.DatetimeIndex):
            if self.start_ is None:
                raise DataValidationError("model has no start date; pass days instead")
            return np.asarray(
                (t - self.start_) / pd.Timedelta(days=1), dtype=np.float64
            )
        return as_float_array(t, "t")

    def predict(self, t: npt.ArrayLike | pd.DatetimeIndex) -> FloatOrArray:
        """Predict rates.

        Parameters
        ----------
        t : array_like or pandas.DatetimeIndex
            Days since the start of decline, or dates (when fitted on a
            date-indexed Series).

        Returns
        -------
        float or numpy.ndarray
            Rate, volume/day.

        Raises
        ------
        FitError
            If the model is not fitted.
        """
        self._check_fitted()
        return arps_rate(self._to_days(t), self.qi_, self.di_, self.b_)

    def cumulative(self, t: npt.ArrayLike | pd.DatetimeIndex) -> FloatOrArray:
        """Cumulative production from the start of decline.

        Parameters
        ----------
        t : array_like or pandas.DatetimeIndex
            Days since the start of decline, or dates.

        Returns
        -------
        float or numpy.ndarray
            Cumulative volume.

        Raises
        ------
        FitError
            If the model is not fitted.
        """
        self._check_fitted()
        return arps_cumulative(self._to_days(t), self.qi_, self.di_, self.b_)

    def time_to_limit(self, econ_limit: float) -> float:
        """Days from the start of decline until the rate reaches ``econ_limit``.

        Parameters
        ----------
        econ_limit : float
            Economic limit rate, volume/day.

        Returns
        -------
        float
            Time in days.

        Raises
        ------
        FitError
            If the model is not fitted.
        """
        self._check_fitted()
        return float(arps_time_to_rate(econ_limit, self.qi_, self.di_, self.b_))

    def eur(self, econ_limit: float, *, max_years: float | None = None) -> float:
        """Estimated ultimate recovery from the start of decline.

        Parameters
        ----------
        econ_limit : float
            Economic limit rate, volume/day.
        max_years : float, optional
            Cap on the producing life after the start of decline, years.

        Returns
        -------
        float
            Cumulative volume from ``t = 0`` to the economic limit. Production
            before the decline start (the ramp-up removed by ``from_peak``)
            is not included.

        Raises
        ------
        FitError
            If the model is not fitted.
        DataValidationError
            If ``econ_limit`` is not positive.
        """
        check_positive(econ_limit, "econ_limit")
        t_end = self.time_to_limit(econ_limit)
        if max_years is not None:
            t_end = min(t_end, max_years * _DAYS_PER_YEAR)
        return float(self.cumulative(t_end))

    def remaining(self, econ_limit: float, *, max_years: float | None = None) -> float:
        """Remaining reserves after the last observation.

        Parameters
        ----------
        econ_limit : float
            Economic limit rate, volume/day.
        max_years : float, optional
            Cap on the producing life after the start of decline, years.

        Returns
        -------
        float
            ``eur(econ_limit) - cumulative(t_last_)``, never negative.
        """
        total = self.eur(econ_limit, max_years=max_years)
        return max(total - float(self.cumulative(self.t_last_)), 0.0)

    def forecast(self, days: int, *, econ_limit: float | None = None) -> pd.DataFrame:
        """Daily forecast after the last observation.

        Parameters
        ----------
        days : int
            Number of days to forecast.
        econ_limit : float, optional
            Stop the forecast when the rate falls below this value.

        Returns
        -------
        pandas.DataFrame
            Columns ``t`` (days since decline start), ``rate`` and
            ``cumulative``; indexed by date when the start date is known.

        Raises
        ------
        FitError
            If the model is not fitted.
        """
        self._check_fitted()
        t = self.t_last_ + np.arange(1, days + 1, dtype=np.float64)
        rate = np.asarray(self.predict(t))
        if econ_limit is not None:
            keep = rate >= econ_limit
            t, rate = t[keep], rate[keep]
        frame = pd.DataFrame(
            {"t": t, "rate": rate, "cumulative": np.asarray(self.cumulative(t))}
        )
        if self.start_ is not None:
            frame.index = pd.DatetimeIndex(
                self.start_ + pd.to_timedelta(t, unit="D"), name="date"
            )
        result: pd.DataFrame = frame
        return result

    # -- uncertainty -------------------------------------------------------------

    def bootstrap(self, n_samples: int = 200, *, seed: int | None = 0) -> DeclineModel:
        """Estimate parameter uncertainty by residual bootstrap.

        Log-rate residuals of the fit are resampled with replacement, added to
        the fitted curve and the selected model is refitted ``n_samples``
        times.

        Parameters
        ----------
        n_samples : int, optional
            Number of bootstrap replicates. Default 200.
        seed : int or None, optional
            Random seed. Default 0.

        Returns
        -------
        DeclineModel
            ``self``, with samples stored for :meth:`predict_interval` and
            :meth:`eur_quantiles`.

        Raises
        ------
        FitError
            If the model was not fitted on data.
        """
        self._check_fitted()
        if not hasattr(self, "_residuals"):
            raise FitError("bootstrap needs a model fitted on data")
        if n_samples < 10:
            raise DataValidationError("n_samples must be >= 10")
        rng = np.random.default_rng(seed)
        fitted = self._log_q - self._residuals
        p0 = [self.qi_, self.di_, self.b_]
        samples = []
        for _ in range(n_samples):
            y = fitted + rng.choice(self._residuals, size=len(fitted), replace=True)
            try:
                f = _fit_one(self.model_, self._t, y, self.b_bounds, self.loss, p0)
            except FitError:  # pragma: no cover - defensive
                continue
            samples.append((f.qi, f.di, f.b))
        self._samples = np.asarray(samples, dtype=np.float64)
        return self

    def _require_samples(self) -> FloatArray:
        if self._samples is None:
            raise FitError("no bootstrap samples; call bootstrap() first")
        return self._samples

    def predict_interval(self, t: npt.ArrayLike | pd.DatetimeIndex) -> pd.DataFrame:
        """P90, P50 and P10 rate bands from the bootstrap samples.

        Parameters
        ----------
        t : array_like or pandas.DatetimeIndex
            Days since decline start, or dates.

        Returns
        -------
        pandas.DataFrame
            Columns ``p90`` (low), ``p50`` and ``p10`` (high), one row per
            time.

        Raises
        ------
        FitError
            If :meth:`bootstrap` has not been called.
        """
        samples = self._require_samples()
        days = np.atleast_1d(self._to_days(t))
        curves = np.vstack([np.asarray(arps_rate(days, *row)) for row in samples])
        low, mid, high = np.percentile(curves, [10, 50, 90], axis=0)
        index = t if isinstance(t, pd.DatetimeIndex) else pd.Index(days, name="t")
        bands: pd.DataFrame = pd.DataFrame(
            {"p90": low, "p50": mid, "p10": high}, index=index
        )
        return bands

    def eur_quantiles(
        self, econ_limit: float, *, max_years: float | None = None
    ) -> pd.Series:
        """P90, P50 and P10 EUR from the bootstrap samples.

        Parameters
        ----------
        econ_limit : float
            Economic limit rate, volume/day.
        max_years : float, optional
            Cap on the producing life after decline start, years.

        Returns
        -------
        pandas.Series
            Index ``p90``, ``p50``, ``p10`` (ascending volumes).

        Raises
        ------
        FitError
            If :meth:`bootstrap` has not been called.
        """
        samples = self._require_samples()
        values = []
        for qi, di, b in samples:
            t_end = float(arps_time_to_rate(econ_limit, qi, di, b))
            if max_years is not None:
                t_end = min(t_end, max_years * _DAYS_PER_YEAR)
            values.append(float(arps_cumulative(t_end, qi, di, b)))
        low, mid, high = np.percentile(values, [10, 50, 90])
        eur: pd.Series = pd.Series({"p90": low, "p50": mid, "p10": high}, name="eur")
        return eur

    # -- reporting ---------------------------------------------------------------

    def summary(self, econ_limit: float | None = None) -> pd.Series:
        """Fitted parameters and diagnostics.

        Parameters
        ----------
        econ_limit : float, optional
            When given, ``eur`` and ``remaining`` are included.

        Returns
        -------
        pandas.Series
            ``model``, ``qi``, ``di`` (nominal, 1/yr), ``de`` (effective
            annual decline, fraction), ``b``, ``start``, ``n_obs``,
            ``residual_std`` (log-rate) and ``aic``, plus ``eur`` and
            ``remaining`` when requested.

        Raises
        ------
        FitError
            If the model is not fitted.
        """
        self._check_fitted()
        de = 1.0 - float(arps_rate(_DAYS_PER_YEAR, 1.0, self.di_, self.b_))
        data: dict[str, Any] = {
            "model": self.model_,
            "qi": self.qi_,
            "di": self.di_,
            "de": de,
            "b": self.b_,
            "start": self.start_,
            "n_obs": self.n_obs_,
            "residual_std": self.residual_std_,
            "aic": self.aic_.get(self.model_, float("nan")),
        }
        if econ_limit is not None:
            data["eur"] = self.eur(econ_limit)
            data["remaining"] = self.remaining(econ_limit)
        summary: pd.Series = pd.Series(data, name="decline")
        return summary

    def plot(
        self,
        rates: pd.Series | None = None,
        *,
        forecast_days: int = 3650,
        title: str | None = None,
    ) -> Any:
        """Plot observed rates, the fitted curve and bootstrap bands.

        Parameters
        ----------
        rates : pandas.Series, optional
            Observed rates to overlay (date-indexed if the model is).
        forecast_days : int, optional
            Forecast horizon after the last observation. Default 3650.
        title : str, optional
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure with a logarithmic rate axis.

        Raises
        ------
        MissingDependencyError
            If plotly is not installed.
        """
        self._check_fitted()
        go = cast(Any, require_extra("plotly.graph_objects", "viz"))
        t = np.linspace(0.0, self.t_last_ + forecast_days, 400)
        x: Any = (
            self.start_ + pd.to_timedelta(t, unit="D") if self.start_ is not None else t
        )
        fig = go.Figure()
        if self._samples is not None:
            bands = self.predict_interval(t)
            fig.add_trace(
                go.Scatter(x=x, y=bands["p10"], name="P10", line={"width": 0})
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=bands["p90"],
                    name="P90 to P10",
                    fill="tonexty",
                    line={"width": 0},
                    fillcolor="rgba(31, 119, 180, 0.2)",
                )
            )
        if rates is not None:
            fig.add_trace(
                go.Scatter(
                    x=rates.index,
                    y=rates,
                    mode="markers",
                    name="observed",
                    marker={"size": 4, "color": "rgba(80, 80, 80, 0.6)"},
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=self.predict(t),
                name=f"{self.model_} fit",
                line={"color": "#1f77b4"},
            )
        )
        fig.update_layout(
            title=title or f"Arps {self.model_} decline (b = {self.b_:.2f})",
            yaxis={"type": "log", "title": "rate"},
            xaxis={"title": "date" if self.start_ is not None else "days"},
        )
        return fig

    def __repr__(self) -> str:
        if not self._fitted:
            return f"DeclineModel(model={self.model!r}, unfitted)"
        return (
            f"DeclineModel({self.model_}, qi={self.qi_:.4g}, di={self.di_:.4g}/yr, "
            f"b={self.b_:.3g})"
        )


# -----------------------------------------------------------------------------
# Batch fitting
# -----------------------------------------------------------------------------


def fit_many(
    df: pd.DataFrame,
    by: str = "well",
    rate_column: str = "oil_rate",
    *,
    model: Literal["auto", "exponential", "hyperbolic", "harmonic"] = "auto",
    econ_limit: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fit a decline model to every well.

    Wells that cannot be fitted are reported with ``status`` set to the error
    message instead of aborting the batch.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table with a ``DatetimeIndex``.
    by : str, optional
        Grouping column. Default ``"well"``.
    rate_column : str, optional
        Rate column to fit. Default ``"oil_rate"``.
    model : str, optional
        Passed to :class:`DeclineModel`. Default ``"auto"``.
    econ_limit : float, optional
        Economic limit for EUR and remaining reserves.
    **kwargs
        Other :class:`DeclineModel` options.

    Returns
    -------
    pandas.DataFrame
        One row per well (index ``by``) with the columns of
        :meth:`DeclineModel.summary` and ``status`` (``"ok"`` or the error).
        Rebuild a model with :meth:`DeclineModel.from_params`.

    Raises
    ------
    DataValidationError
        If ``by`` or ``rate_column`` is missing or the index is not a
        ``DatetimeIndex``.

    Examples
    --------
    >>> from wellcast.datasets import make_synthetic_field
    >>> field = make_synthetic_field(n_wells=2, n_injectors=0)
    >>> table = fit_many(field.production, econ_limit=20.0)
    >>> table["status"].tolist()
    ['ok', 'ok']
    """
    for column in (by, rate_column):
        if column not in df.columns:
            raise DataValidationError(f"missing required column: {column}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("a DatetimeIndex is required")
    rows: dict[str, pd.Series] = {}
    for name, group in df.groupby(by, sort=True):
        key = str(name)
        try:
            fitted = DeclineModel(model, **kwargs).fit(group[rate_column])
            row = fitted.summary(econ_limit)
            row["status"] = "ok"
        except (FitError, DataValidationError) as exc:
            row = pd.Series({"status": str(exc)})
        rows[key] = row
    table: pd.DataFrame = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = by
    return table


# =============================================================================
# End of module: wellcast.dca
# =============================================================================
