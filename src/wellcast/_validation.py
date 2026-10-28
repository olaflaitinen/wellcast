# =============================================================================
# wellcast: shared input validation and optional-import helpers
# -----------------------------------------------------------------------------
# Module:      wellcast._validation
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

"""Private helpers for validating inputs and importing optional extras."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterable
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from wellcast.exceptions import DataValidationError, MissingDependencyError

__all__ = [
    "RangeWarning",
    "as_float_array",
    "check_fraction",
    "check_non_negative",
    "check_positive",
    "require_columns",
    "require_extra",
    "warn_outside_range",
]


class RangeWarning(UserWarning):
    """An input lies outside the range a correlation was developed for."""


# -----------------------------------------------------------------------------
# Array coercion and numeric checks
# -----------------------------------------------------------------------------


def as_float_array(value: Any, name: str) -> npt.NDArray[np.float64]:
    """Convert a scalar or array-like to a float64 array.

    Parameters
    ----------
    value : array_like
        Scalar, sequence, ``numpy`` array or ``pandas`` Series.
    name : str
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Float64 array (zero-dimensional for scalar input).

    Raises
    ------
    DataValidationError
        If the value cannot be interpreted as numbers.

    Examples
    --------
    >>> as_float_array([1, 2], "q")
    array([1., 2.])
    """
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be numeric, got {value!r}") from exc


def check_positive(value: Any, name: str) -> npt.NDArray[np.float64]:
    """Return ``value`` as a float array, requiring every finite entry to be > 0.

    NaN entries are allowed and passed through, so that gaps in field data do
    not abort a vectorized calculation.

    Parameters
    ----------
    value : array_like
        Values to check.
    name : str
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        The validated float64 array.

    Raises
    ------
    DataValidationError
        If any non-NaN entry is zero or negative.

    Examples
    --------
    >>> check_positive(3.0, "pressure")
    array(3.)
    """
    arr = as_float_array(value, name)
    if np.any(arr[~np.isnan(arr)] <= 0.0):
        raise DataValidationError(f"{name} must be strictly positive")
    return arr


def check_non_negative(value: Any, name: str) -> npt.NDArray[np.float64]:
    """Return ``value`` as a float array, requiring every finite entry to be >= 0.

    Parameters
    ----------
    value : array_like
        Values to check.
    name : str
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        The validated float64 array.

    Raises
    ------
    DataValidationError
        If any non-NaN entry is negative.

    Examples
    --------
    >>> check_non_negative([0.0, 5.0], "rate")
    array([0., 5.])
    """
    arr = as_float_array(value, name)
    if np.any(arr[~np.isnan(arr)] < 0.0):
        raise DataValidationError(f"{name} must be non-negative")
    return arr


def check_fraction(value: Any, name: str) -> npt.NDArray[np.float64]:
    """Return ``value`` as a float array, requiring entries in the interval [0, 1].

    Parameters
    ----------
    value : array_like
        Values to check, for example a water cut.
    name : str
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        The validated float64 array.

    Raises
    ------
    DataValidationError
        If any non-NaN entry is outside [0, 1].

    Examples
    --------
    >>> check_fraction(0.25, "water_cut")
    array(0.25)
    """
    arr = as_float_array(value, name)
    finite = arr[~np.isnan(arr)]
    if np.any((finite < 0.0) | (finite > 1.0)):
        raise DataValidationError(f"{name} must lie in the interval [0, 1]")
    return arr


def warn_outside_range(
    value: Any,
    name: str,
    low: float,
    high: float,
    correlation: str,
) -> None:
    """Warn when any entry lies outside the published range of a correlation.

    Empirical correlations (Standing, Vasquez-Beggs and others) are fitted to
    a finite data set. Extrapolating beyond it is allowed but flagged.

    Parameters
    ----------
    value : array_like
        Values to check.
    name : str
        Argument name used in the warning.
    low, high : float
        Inclusive bounds of the correlation's data set.
    correlation : str
        Correlation name used in the warning.

    Warns
    -----
    RangeWarning
        If any non-NaN entry is outside ``[low, high]``.

    Examples
    --------
    >>> warn_outside_range(150.0, "temperature", 100.0, 258.0, "Standing")
    """
    arr = as_float_array(value, name)
    finite = arr[~np.isnan(arr)]
    if finite.size and (np.min(finite) < low or np.max(finite) > high):
        warnings.warn(
            f"{name} outside the {correlation} range [{low:g}, {high:g}]; "
            "results are extrapolated",
            RangeWarning,
            stacklevel=3,
        )


# -----------------------------------------------------------------------------
# DataFrame checks
# -----------------------------------------------------------------------------


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise if any of ``columns`` is missing from ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to inspect.
    columns : iterable of str
        Required column names.

    Raises
    ------
    DataValidationError
        Listing every missing column.

    Examples
    --------
    >>> import pandas as pd
    >>> require_columns(pd.DataFrame({"oil_rate": [1.0]}), ["oil_rate"])
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing required column(s): {', '.join(missing)}")


# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------


def require_extra(module: str, extra: str) -> ModuleType:
    """Import an optional dependency or fail with an install hint.

    Parameters
    ----------
    module : str
        Importable module name, for example ``"lightgbm"``.
    extra : str
        wellcast extra that provides it, for example ``"ml"``.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    MissingDependencyError
        If the module cannot be imported.

    Examples
    --------
    >>> require_extra("numpy", "core").__name__
    'numpy'
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(module.split(".")[0], extra) from exc


# =============================================================================
# End of module: wellcast._validation
# =============================================================================
