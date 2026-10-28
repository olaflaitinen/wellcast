# =============================================================================
# wellcast: production data and LAS log ingestion into a canonical schema
# -----------------------------------------------------------------------------
# Module:      wellcast.io
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

"""Read field production data and well logs into standard DataFrames.

Canonical production schema
---------------------------
:func:`read_production` returns a DataFrame with a ``DatetimeIndex`` named
``date``, sorted by well and date, and the columns below. Only ``well`` is
mandatory; the others are present when the source provides them. Columns that
do not match the schema are kept unchanged after the canonical ones.

===================  ======  =======  ====================================
Column               Unit    dtype    Description
===================  ======  =======  ====================================
``well``             -       str      Well identifier
``oil_rate``         bbl/d   float64  Oil rate at stock-tank conditions
``gas_rate``         Mscf/d  float64  Gas rate at standard conditions
``water_rate``       bbl/d   float64  Water rate
``water_inj_rate``   bbl/d   float64  Water injection rate
``gas_inj_rate``     Mscf/d  float64  Gas injection or gas lift rate
``whp``              psi     float64  Wellhead (tubing head) pressure
``wht``              degF    float64  Wellhead temperature
``bhp``              psi     float64  Bottomhole (downhole gauge) pressure
``bht``              degF    float64  Bottomhole temperature
``choke``            in/64   float64  Choke bean size
``on_stream_hours``  h       float64  Producing hours in the period
===================  ======  =======  ====================================
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from wellcast._validation import require_extra
from wellcast.exceptions import DataValidationError
from wellcast.units import convert

__all__ = [
    "CANONICAL_SCHEMA",
    "DEFAULT_ALIASES",
    "ColumnSpec",
    "from_dataframe",
    "read_las",
    "read_production",
    "standardize_columns",
]


@dataclass(frozen=True)
class ColumnSpec:
    """Description of one column in the canonical production schema.

    Parameters
    ----------
    name : str
        Canonical column name.
    unit : str
        Unit symbol understood by :func:`wellcast.units.convert`, or ``"-"``.
    dtype : str
        Expected dtype.
    description : str
        Human-readable description.
    """

    name: str
    unit: str
    dtype: str
    description: str


# -----------------------------------------------------------------------------
# Canonical schema and alias map
# -----------------------------------------------------------------------------

CANONICAL_SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec("well", "-", "str", "Well identifier"),
    ColumnSpec("oil_rate", "bbl/d", "float64", "Oil rate at stock-tank conditions"),
    ColumnSpec("gas_rate", "Mscf/d", "float64", "Gas rate at standard conditions"),
    ColumnSpec("water_rate", "bbl/d", "float64", "Water rate"),
    ColumnSpec("water_inj_rate", "bbl/d", "float64", "Water injection rate"),
    ColumnSpec("gas_inj_rate", "Mscf/d", "float64", "Gas injection or gas lift rate"),
    ColumnSpec("whp", "psi", "float64", "Wellhead (tubing head) pressure"),
    ColumnSpec("wht", "degF", "float64", "Wellhead temperature"),
    ColumnSpec("bhp", "psi", "float64", "Bottomhole (downhole gauge) pressure"),
    ColumnSpec("bht", "degF", "float64", "Bottomhole temperature"),
    ColumnSpec("choke", "in/64", "float64", "Choke bean size"),
    ColumnSpec("on_stream_hours", "h", "float64", "Producing hours in the period"),
)

_SCHEMA = {spec.name: spec for spec in CANONICAL_SCHEMA}

# Keys are normalized with _normalize (upper case, non-alphanumerics removed),
# so "Oil Rate", "oil_rate" and "OIL-RATE" all match OILRATE.
DEFAULT_ALIASES: dict[str, str] = {
    # Date
    "DATE": "date",
    "DATEPRD": "date",
    "DATETIME": "date",
    "TIME": "date",
    "TIMESTAMP": "date",
    "PRODDATE": "date",
    "PRODUCTIONDATE": "date",
    # Well identifier
    "WELL": "well",
    "WELLNAME": "well",
    "WELLID": "well",
    "WELLBORE": "well",
    "WELLBORENAME": "well",
    "WELLBORECODE": "well",
    "NPDWELLBORENAME": "well",
    "UWI": "well",
    "API": "well",
    # Rates
    "OILRATE": "oil_rate",
    "OIL": "oil_rate",
    "QO": "oil_rate",
    "BOREOILVOL": "oil_rate",
    "GASRATE": "gas_rate",
    "GAS": "gas_rate",
    "QG": "gas_rate",
    "BOREGASVOL": "gas_rate",
    "WATERRATE": "water_rate",
    "WATER": "water_rate",
    "QW": "water_rate",
    "BOREWATVOL": "water_rate",
    "WATERINJRATE": "water_inj_rate",
    "WATERINJECTION": "water_inj_rate",
    "BOREWIVOL": "water_inj_rate",
    "GASINJRATE": "gas_inj_rate",
    "GASLIFTRATE": "gas_inj_rate",
    "GASINJECTION": "gas_inj_rate",
    # Pressures and temperatures
    "WHP": "whp",
    "THP": "whp",
    "FTHP": "whp",
    "AVGWHPP": "whp",
    "WELLHEADPRESSURE": "whp",
    "TUBINGHEADPRESSURE": "whp",
    "WHT": "wht",
    "AVGWHTP": "wht",
    "WELLHEADTEMPERATURE": "wht",
    "BHP": "bhp",
    "PWF": "bhp",
    "AVGDOWNHOLEPRESSURE": "bhp",
    "BOTTOMHOLEPRESSURE": "bhp",
    "DOWNHOLEPRESSURE": "bhp",
    "BHT": "bht",
    "AVGDOWNHOLETEMPERATURE": "bht",
    "BOTTOMHOLETEMPERATURE": "bht",
    # Choke and uptime
    "CHOKE": "choke",
    "CHOKESIZE": "choke",
    "AVGCHOKESIZEP": "choke",
    "ONSTREAMHOURS": "on_stream_hours",
    "ONSTREAMHRS": "on_stream_hours",
    "HOURSON": "on_stream_hours",
    "UPTIME": "on_stream_hours",
}

_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
_TEXT_SUFFIXES = {".csv", ".txt"}


def _normalize(name: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(name).upper())


# -----------------------------------------------------------------------------
# Column standardization
# -----------------------------------------------------------------------------


def standardize_columns(
    df: pd.DataFrame,
    alias_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Rename source columns to canonical names.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw table.
    alias_map : mapping of str to str, optional
        Extra aliases ``{source_name: canonical_name}``. They take precedence
        over :data:`DEFAULT_ALIASES`. Matching ignores case and punctuation.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with renamed columns. When two source columns map to
        the same canonical name, the first one wins and the other keeps its
        original name.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({"DATEPRD": [], "BORE_OIL_VOL": [], "AVG_WHP_P": []})
    >>> list(standardize_columns(raw).columns)
    ['date', 'oil_rate', 'whp']
    """
    aliases = dict(DEFAULT_ALIASES)
    if alias_map:
        aliases.update({_normalize(k): v for k, v in alias_map.items()})
    renames: dict[str, str] = {}
    taken: set[str] = set()
    for column in df.columns:
        target = aliases.get(_normalize(column))
        if target is not None and target not in taken:
            renames[column] = target
            taken.add(target)
    renamed: pd.DataFrame = df.rename(columns=renames)
    return renamed


# -----------------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------------


def _read_table(path: Path, sheet_name: str | int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return pd.read_csv(path)
    if suffix in _EXCEL_SUFFIXES:
        require_extra("openpyxl", "io")
        return pd.read_excel(path, sheet_name=sheet_name)
    raise DataValidationError(
        f"unsupported file type {suffix!r}; expected one of "
        f"{sorted(_TEXT_SUFFIXES | _EXCEL_SUFFIXES)}"
    )


def read_production(
    path: str | PathLike[str],
    alias_map: Mapping[str, str] | None = None,
    *,
    units: Mapping[str, str] | None = None,
    well: str | None = None,
    sheet_name: str | int = 0,
    dayfirst: bool = False,
) -> pd.DataFrame:
    r"""Read a CSV or Excel production table into the canonical schema.

    Parameters
    ----------
    path : str or path-like
        ``.csv``/``.txt`` or ``.xlsx``/``.xlsm``/``.xls`` file.
    alias_map : mapping of str to str, optional
        Extra column aliases ``{source_name: canonical_name}``.
    units : mapping of str to str, optional
        Units of canonical columns in the source, for example
        ``{"oil_rate": "sm3/d", "whp": "bar"}``. Those columns are converted
        to the canonical units listed in :data:`CANONICAL_SCHEMA`.
    well : str, optional
        Well name to use when the file has no well column.
    sheet_name : str or int, optional
        Excel sheet. Default is the first sheet.
    dayfirst : bool, optional
        Parse ambiguous dates as day first (``31/01/2024``).

    Returns
    -------
    pandas.DataFrame
        Table indexed by a ``DatetimeIndex`` named ``date``.

    Raises
    ------
    DataValidationError
        If the file type is unsupported, no date column is found, dates cannot
        be parsed, or no well identifier is available.
    MissingDependencyError
        If an Excel file is read without ``openpyxl`` installed.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from wellcast.io import read_production
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "prod.csv"
    >>> _ = tmp.write_text("DATEPRD,WELL,BORE_OIL_VOL\n2024-01-01,A-1,100\n")
    >>> df = read_production(tmp, units={"oil_rate": "m3/d"})
    >>> round(float(df["oil_rate"].iloc[0]), 1)
    629.0
    """
    raw = _read_table(Path(path), sheet_name)
    return from_dataframe(raw, alias_map, units=units, well=well, dayfirst=dayfirst)


def from_dataframe(
    raw: pd.DataFrame,
    alias_map: Mapping[str, str] | None = None,
    *,
    units: Mapping[str, str] | None = None,
    well: str | None = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Convert an in-memory table into the canonical schema.

    Same processing as :func:`read_production`, for data that is already in
    a DataFrame (for example from a database query or a web upload).

    Parameters
    ----------
    raw : pandas.DataFrame
        Source table with a date column; its index is ignored.
    alias_map : mapping of str to str, optional
        Extra column aliases ``{source_name: canonical_name}``.
    units : mapping of str to str, optional
        Source units of canonical columns, converted to canonical units.
    well : str, optional
        Well name to use when the table has no well column.
    dayfirst : bool, optional
        Parse ambiguous dates as day first.

    Returns
    -------
    pandas.DataFrame
        Table indexed by a ``DatetimeIndex`` named ``date``, sorted by well
        and date.

    Raises
    ------
    DataValidationError
        If no date column is found, dates cannot be parsed, or no well
        identifier is available.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({"Date": ["2024-01-02", "2024-01-01"], "Qo": [90, 100]})
    >>> from_dataframe(raw, well="A")["oil_rate"].tolist()
    [100.0, 90.0]
    """
    standardized = standardize_columns(raw.reset_index(drop=True), alias_map)
    return _finalize(standardized, units, well, dayfirst)


def _finalize(
    df: pd.DataFrame,
    units: Mapping[str, str] | None,
    well: str | None,
    dayfirst: bool,
) -> pd.DataFrame:
    if "date" not in df.columns:
        raise DataValidationError(
            "no date column found; pass alias_map={'<your column>': 'date'}"
        )
    try:
        dates = pd.to_datetime(df["date"], dayfirst=dayfirst, format="mixed")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"could not parse dates: {exc}") from exc
    df = df.drop(columns="date")
    if "well" not in df.columns:
        if well is None:
            raise DataValidationError(
                "no well column found; pass well='<name>' for single-well files"
            )
        df.insert(0, "well", well)
    df["well"] = df["well"].astype(str)

    for name in _SCHEMA:
        if name != "well" and name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("float64")

    for name, unit in (units or {}).items():
        if name not in _SCHEMA or name == "well":
            raise DataValidationError(f"{name!r} is not a numeric canonical column")
        if name in df.columns:
            df[name] = convert(df[name], unit, _SCHEMA[name].unit)

    df.index = pd.DatetimeIndex(dates, name="date")
    canonical = [n for n in _SCHEMA if n in df.columns]
    others = [c for c in df.columns if c not in _SCHEMA]
    df = df[canonical + others]
    order = df.reset_index().sort_values(["well", "date"], kind="stable").index
    result: pd.DataFrame = df.iloc[order]
    return result


def read_las(path: str | PathLike[str]) -> pd.DataFrame:
    """Read a LAS well log file into a depth-indexed DataFrame.

    Parameters
    ----------
    path : str or path-like
        LAS 1.2, 2.0 or 3.0 file.

    Returns
    -------
    pandas.DataFrame
        Curves as columns, indexed by the first (depth) curve. Header items
        from the ``~Well`` section are stored in ``df.attrs["well"]`` and curve
        units in ``df.attrs["units"]``.

    Raises
    ------
    MissingDependencyError
        If ``lasio`` is not installed.

    Examples
    --------
    >>> import tempfile, pathlib, numpy as np
    >>> import lasio
    >>> las = lasio.LASFile()
    >>> las.append_curve("DEPT", np.array([1000.0, 1000.5]), unit="ft")
    >>> las.append_curve("GR", np.array([45.0, 60.0]), unit="API")
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "log.las"
    >>> las.write(str(tmp))
    >>> read_las(tmp)["GR"].tolist()
    [45.0, 60.0]
    """
    lasio: Any = require_extra("lasio", "io")
    las = lasio.read(str(path))
    df: pd.DataFrame = las.df()
    df.attrs["well"] = {item.mnemonic: item.value for item in las.well}
    df.attrs["units"] = {curve.mnemonic: curve.unit for curve in las.curves}
    return df


# =============================================================================
# End of module: wellcast.io
# =============================================================================
