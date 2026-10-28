# =============================================================================
# wellcast: tests for the synthetic field generator
# -----------------------------------------------------------------------------
# Module:      tests/test_datasets.py
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

import numpy as np
import pandas as pd
import pytest

from wellcast.clean import detect_shut_ins, hampel_filter
from wellcast.datasets import ANOMALY_TYPES, SyntheticField, make_synthetic_field
from wellcast.exceptions import DataValidationError
from wellcast.io import CANONICAL_SCHEMA


@pytest.fixture(scope="module")
def field() -> SyntheticField:
    return make_synthetic_field()


def _producer(field: SyntheticField, well: str = "P-01") -> pd.DataFrame:
    return field.production[field.production["well"] == well]


# -----------------------------------------------------------------------------
# Structure
# -----------------------------------------------------------------------------


def test_defaults_shape_and_schema(field: SyntheticField) -> None:
    prod = field.production
    assert isinstance(prod.index, pd.DatetimeIndex)
    assert prod.index.name == "date"
    # Every canonical column except gas injection, in schema order.
    expected = [s.name for s in CANONICAL_SCHEMA if s.name != "gas_inj_rate"]
    assert list(prod.columns) == expected
    assert sorted(prod["well"].unique()) == [
        "I-01",
        "P-01",
        "P-02",
        "P-03",
        "P-04",
        "P-05",
    ]
    assert prod.index.min() >= pd.Timestamp("2020-01-01")
    assert prod.index.max() < pd.Timestamp("2025-01-01")
    # Sorted by well then date, with a regular daily grid per well.
    for _, g in prod.groupby("well"):
        assert g.index.is_monotonic_increasing
        assert (g.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_labels_align(field: SyntheticField) -> None:
    labels = field.labels
    assert labels.index.equals(field.production.index)
    assert (labels["well"].to_numpy() == field.production["well"].to_numpy()).all()
    assert list(labels.columns) == [
        "well",
        "shut_in",
        "anomaly",
        "anomaly_type",
        "water_breakthrough",
    ]
    assert (labels["anomaly"] == (labels["anomaly_type"] != "")).all()


def test_wells_table(field: SyntheticField) -> None:
    wells = field.wells
    assert wells.index.name == "well"
    assert set(wells.loc[["P-01", "P-05"], "type"]) == {"producer"}
    assert wells.loc["I-01", "type"] == "injector"
    producers = wells[wells["type"] == "producer"]
    assert (producers["bubble_point"] < producers["initial_pressure"]).all()
    assert ((producers["b"] > 0) & (producers["b"] < 1)).all()


def test_reproducible_and_seed_dependent() -> None:
    a = make_synthetic_field(n_wells=2, years=1, seed=7)
    b = make_synthetic_field(n_wells=2, years=1, seed=7)
    c = make_synthetic_field(n_wells=2, years=1, seed=8)
    pd.testing.assert_frame_equal(a.production, b.production)
    assert not a.production["oil_rate"].equals(c.production["oil_rate"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_wells": 0}, "n_wells"),
        ({"n_injectors": -1}, "n_injectors"),
        ({"years": 0}, "years"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(DataValidationError, match=match):
        make_synthetic_field(**kwargs)


def test_weekly_frequency() -> None:
    daily = make_synthetic_field(n_wells=2, years=2, n_injectors=0)
    weekly = make_synthetic_field(n_wells=2, years=2, n_injectors=0, freq="W")
    assert len(weekly.production) < len(daily.production) / 6
    assert weekly.labels.index.equals(weekly.production.index)
    assert weekly.labels["anomaly"].any()
    assert weekly.labels["shut_in"].dtype == bool
    kinds = {k for v in weekly.labels["anomaly_type"] for k in v.split(",") if k}
    assert kinds == set(ANOMALY_TYPES)


# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------


def test_physical_bounds(field: SyntheticField) -> None:
    prod = field.production
    clean = prod[~field.labels["anomaly"]]
    for column in (
        "oil_rate",
        "gas_rate",
        "water_rate",
        "water_inj_rate",
        "whp",
        "bhp",
    ):
        assert (clean[column].dropna() >= 0).all(), column
    # Wellhead pressure never exceeds bottomhole pressure.
    both = clean[["whp", "bhp"]].dropna()
    assert (both["whp"] < both["bhp"]).all()
    assert prod["on_stream_hours"].between(0, 24).all()


def test_decline_trend(field: SyntheticField) -> None:
    for well in ("P-01", "P-02", "P-03"):
        oil = _producer(field, well)["oil_rate"]
        first = oil.iloc[:60].median()
        last = oil.iloc[-60:].median()
        assert last < 0.5 * first


def test_water_breakthrough(field: SyntheticField) -> None:
    prod = _producer(field)
    labels = field.labels.loc[field.labels["well"] == "P-01"]
    bt = field.wells.loc["P-01", "breakthrough_date"]
    assert (prod.loc[prod.index < bt, "water_rate"] == 0).all()
    assert (labels["water_breakthrough"] == (labels.index >= bt)).all()
    late = prod.iloc[-90:]
    assert (late["water_rate"] / (late["water_rate"] + late["oil_rate"])).median() > 0.2


def test_gor_rises_below_bubble_point(field: SyntheticField) -> None:
    prod = _producer(field)
    flowing = prod[prod["oil_rate"] > 0]
    gor = 1000.0 * flowing["gas_rate"] / flowing["oil_rate"]
    rs = field.wells.loc["P-01", "rs"]
    assert gor.iloc[:30].median() == pytest.approx(rs, rel=0.1)
    assert gor.iloc[-60:].median() > 1.2 * rs


def test_shut_ins_match_labels(field: SyntheticField) -> None:
    flags = detect_shut_ins(field.production)
    np.testing.assert_array_equal(flags.to_numpy(), field.labels["shut_in"].to_numpy())


def test_shut_in_pressure_builds_up(field: SyntheticField) -> None:
    prod = _producer(field)
    shut = field.labels.loc[field.labels["well"] == "P-01", "shut_in"].to_numpy()
    start = int(np.argmax(shut))
    assert prod["bhp"].iloc[start + 1] > prod["bhp"].iloc[start - 1]
    assert prod["oil_rate"].iloc[start] == 0.0


def test_injector_loses_injectivity(field: SyntheticField) -> None:
    inj = field.production[field.production["well"] == "I-01"]
    loss = field.wells.loc["I-01", "injectivity_loss_date"]
    flowing = inj[inj["on_stream_hours"] > 0]
    pr = field.wells.loc["I-01", "initial_pressure"]
    ii = flowing["water_inj_rate"] / (flowing["bhp"] - pr)
    before = ii[ii.index < loss].median()
    after = ii[ii.index >= loss].median()
    assert after == pytest.approx(0.6 * before, rel=0.05)
    assert inj[["oil_rate", "gas_rate", "water_rate"]].isna().all().all()


def test_spikes_are_detectable(field: SyntheticField) -> None:
    labels = field.labels
    spikes = labels["anomaly_type"] == "spike"
    assert spikes.groupby(labels["well"]).sum().loc["P-01":"P-05"].eq(1).all()
    prod = _producer(field)
    flags = hampel_filter(prod["oil_rate"], window=7)
    spike_day = labels.loc[(labels["well"] == "P-01") & spikes].index[0]
    assert flags.loc[spike_day]


# =============================================================================
# End of module: tests/test_datasets.py
# =============================================================================
