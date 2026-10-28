# =============================================================================
# wellcast: tests for production and LAS readers
# -----------------------------------------------------------------------------
# Module:      tests/test_io.py
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

from __future__ import annotations

import sys
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import pytest

from wellcast.exceptions import DataValidationError, MissingDependencyError
from wellcast.io import (
    CANONICAL_SCHEMA,
    DEFAULT_ALIASES,
    read_las,
    read_production,
    standardize_columns,
)

RAW = pd.DataFrame(
    {
        "DATEPRD": ["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02"],
        "NPD_WELL_BORE_NAME": ["B", "B", "A", "A"],
        "BORE_OIL_VOL": [50.0, 55.0, 100.0, "n/a"],
        "AVG_WHP_P": [20.0, 21.0, 30.0, 31.0],
        "AVG_DOWNHOLE_PRESSURE": [200.0, 201.0, 210.0, 211.0],
        "FLOW_KIND": ["production"] * 4,
    }
)


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "prod.csv"
    RAW.to_csv(path, index=False)
    return path


def test_schema_and_aliases_are_consistent() -> None:
    names = {spec.name for spec in CANONICAL_SCHEMA}
    assert set(DEFAULT_ALIASES.values()) <= names | {"date"}


def test_read_csv_standardizes(csv_path: Path) -> None:
    df = read_production(csv_path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert list(df.columns) == ["well", "oil_rate", "whp", "bhp", "FLOW_KIND"]
    # Sorted by well, then date.
    assert df["well"].tolist() == ["A", "A", "B", "B"]
    assert df.index[0] < df.index[1]
    # Unparseable numbers become NaN rather than failing the whole file.
    assert np.isnan(df["oil_rate"].iloc[1])
    assert df["oil_rate"].dtype == np.float64


def test_unit_conversion_on_read(csv_path: Path) -> None:
    df = read_production(csv_path, units={"whp": "bar", "bhp": "bar"})
    assert df.loc[df["well"] == "A", "whp"].iloc[0] == pytest.approx(30.0 * 14.5037738)


def test_unit_for_unknown_column_rejected(csv_path: Path) -> None:
    with pytest.raises(DataValidationError, match="FLOW_KIND"):
        read_production(csv_path, units={"FLOW_KIND": "bar"})


def test_unit_for_absent_canonical_column_is_ignored(csv_path: Path) -> None:
    df = read_production(csv_path, units={"gas_rate": "sm3/d"})
    assert "gas_rate" not in df.columns


def test_read_excel(tmp_path: Path) -> None:
    path = tmp_path / "prod.xlsx"
    RAW.to_excel(path, index=False)
    df = read_production(path)
    assert len(df) == 4
    assert set(df["well"]) == {"A", "B"}


def test_excel_without_openpyxl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(MissingDependencyError, match=r"wellcast\[io\]"):
        read_production(tmp_path / "prod.xlsx")


def test_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="unsupported"):
        read_production(tmp_path / "prod.parquet")


def test_custom_alias_and_single_well(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("Fecha,Petroleo\n01/02/2024,10\n02/02/2024,9\n")
    df = read_production(
        path,
        alias_map={"Fecha": "date", "Petroleo": "oil_rate"},
        well="W-1",
        dayfirst=True,
    )
    assert df["well"].unique().tolist() == ["W-1"]
    assert df.index[1] == pd.Timestamp("2024-02-02")


def test_missing_date(tmp_path: Path) -> None:
    path = tmp_path / "nodate.csv"
    path.write_text("well,oil_rate\nA,1\n")
    with pytest.raises(DataValidationError, match="no date column"):
        read_production(path)


def test_missing_well(tmp_path: Path) -> None:
    path = tmp_path / "nowell.csv"
    path.write_text("date,oil_rate\n2024-01-01,1\n")
    with pytest.raises(DataValidationError, match="no well column"):
        read_production(path)


def test_bad_dates(tmp_path: Path) -> None:
    path = tmp_path / "baddate.csv"
    path.write_text("date,well\nnot-a-date,A\n")
    with pytest.raises(DataValidationError, match="could not parse dates"):
        read_production(path)


def test_duplicate_alias_first_wins() -> None:
    raw = pd.DataFrame({"WHP": [1.0], "THP": [2.0]})
    out = standardize_columns(raw)
    assert list(out.columns) == ["whp", "THP"]


def test_read_las(tmp_path: Path) -> None:
    las = lasio.LASFile()
    las.well["WELL"].value = "A-1"
    las.append_curve("DEPT", np.array([1000.0, 1000.5, 1001.0]), unit="ft")
    las.append_curve("GR", np.array([45.0, 60.0, 52.0]), unit="API")
    path = tmp_path / "log.las"
    las.write(str(path))
    df = read_las(path)
    assert df.index.name == "DEPT"
    assert df.attrs["well"]["WELL"] == "A-1"
    assert df.attrs["units"]["GR"] == "API"


# =============================================================================
# End of module: tests/test_io.py
# =============================================================================
