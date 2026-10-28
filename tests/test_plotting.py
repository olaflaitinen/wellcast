# =============================================================================
# wellcast: tests for the plotly chart helpers
# -----------------------------------------------------------------------------
# Module:      tests/test_plotting.py
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

import numpy as np
import pandas as pd
import pytest

from wellcast import plotting as pl
from wellcast.datasets import make_synthetic_field
from wellcast.dca import DeclineModel
from wellcast.diagnostics import (
    chan_plot_data,
    detect_anomalies,
    hall_plot_data,
    hall_slope_change,
)
from wellcast.exceptions import DataValidationError, MissingDependencyError
from wellcast.ipr import ipr_curve


@pytest.fixture(scope="module")
def production() -> pd.DataFrame:
    return make_synthetic_field(n_wells=2, years=3).production


def test_theme_registration() -> None:
    import plotly.io as pio

    assert pl.register_theme() == "wellcast"
    template = pio.templates["wellcast"]
    assert list(template.layout.colorway) == list(pl.SERIES_COLORS)


def test_surfaces_are_set_on_the_figure() -> None:
    # Renderers such as Streamlit override template-only backgrounds.
    fig = pl.ipr_plot(ipr_curve("vogel", 2500.0, qmax=1000.0))
    assert fig.layout.plot_bgcolor == fig.layout.paper_bgcolor == "#fcfcfb"
    history = pl.production_history(make_synthetic_field(n_wells=1, years=1).production)
    assert history.layout.plot_bgcolor == "#fcfcfb"


def test_fluid_identity_has_color_and_dash() -> None:
    dashes = {
        style["dash"]
        for name, style in pl.FLUID_STYLES.items()
        if name != "liquid_rate"
    }
    colors = {style["color"] for style in pl.FLUID_STYLES.values()}
    assert len(colors) == len(pl.FLUID_STYLES)
    assert len(dashes) >= 2


def test_production_history_uses_stacked_panels(production: pd.DataFrame) -> None:
    fig = pl.production_history(production, "P-02")
    assert fig.layout.title.text == "Production history P-02"
    # Two separate y axes in two rows, never a secondary axis on one panel.
    assert fig.layout.yaxis.title.text == "rate"
    assert fig.layout.yaxis2.title.text == "pressure (psi)"
    oil = next(t for t in fig.data if t.name.startswith("oil"))
    assert oil.line.color == pl.FLUID_STYLES["oil_rate"]["color"]
    assert oil.line.dash == "solid"
    rates_only = pl.production_history(production[["well", "oil_rate"]])
    assert len(rates_only.data) == 1


def test_production_history_without_well_column(production: pd.DataFrame) -> None:
    single = production[production["well"] == "P-01"].drop(columns="well")
    fig = pl.production_history(single, pressures=())
    assert fig.layout.title.text == "Production history"


def test_decline_plot_with_bands_and_limit(production: pd.DataFrame) -> None:
    rates = production.loc[production["well"] == "P-01", "oil_rate"]
    model = DeclineModel().fit(rates).bootstrap(20)
    fig = model.plot(rates, econ_limit=25.0)
    names = [t.name for t in fig.data]
    assert "P90 to P10" in names
    assert names[-1].endswith("fit")
    assert fig.layout.yaxis.type == "log"
    # The forecast stops at the economic limit.
    assert min(fig.data[-1].y) == pytest.approx(25.0, rel=0.02)
    assert any("economic limit" in a.text for a in fig.layout.annotations)
    bare = pl.decline_plot(DeclineModel.from_params(500.0, 0.5, 0.5))
    assert bare.layout.xaxis.title.text == "days"


def test_ipr_plot_single_and_multiple() -> None:
    curve = ipr_curve("vogel", 2500.0, qmax=1000.0)
    single = pl.ipr_plot(curve, operating_point=(350.0, 2000.0))
    assert [t.name for t in single.data] == ["IPR", "test point"]
    multi = pl.ipr_plot({"a": curve, "b": curve})
    assert [t.line.color for t in multi.data] == list(pl.SERIES_COLORS[:2])
    with pytest.raises(DataValidationError):
        pl.ipr_plot(pd.DataFrame({"q": [1.0]}))


def test_predicted_vs_actual_caps_colored_series() -> None:
    y = pd.Series(np.arange(12.0))
    few = pl.predicted_vs_actual(y, y + 1, pd.Series(["A", "B", "C"] * 4))
    assert [t.name for t in few.data] == ["A", "B", "C", "1:1"]
    many = pl.predicted_vs_actual(y, y + 1, pd.Series(list("ABCDEF") * 2))
    assert [t.name for t in many.data] == ["observations", "1:1"]
    assert many.data[0].hovertemplate is not None
    plain = pl.predicted_vs_actual(y, y)
    assert plain.data[0].hovertemplate is None
    nan_only = pl.predicted_vs_actual(pd.Series([np.nan]), pd.Series([np.nan]))
    assert list(nan_only.data[-1].x) == [0.0, 1.0]


def test_chan_hall_and_anomaly_plots(production: pd.DataFrame) -> None:
    chan = chan_plot_data(production[production["well"].str.startswith("P")])
    fig = pl.chan_plot(chan, "P-01")
    assert fig.layout.xaxis.type == "log"
    assert [t.name for t in fig.data] == ["WOR", "WOR'"]
    injector = production[production["well"] == "I-01"]
    hall = hall_plot_data(injector, reservoir_pressure=3000.0)
    hall_fig = pl.hall_plot(hall, hall_slope_change(hall))
    assert "slope" in hall_fig.layout.annotations[0].text
    assert len(pl.hall_plot(hall).layout.shapes) == 0
    anomalies = detect_anomalies(production, method="rules")
    overlay = pl.anomaly_overlay(production, anomalies, "bhp", "P-01")
    expected = int((anomalies["well"] == "P-01").sum())
    assert len(overlay.layout.shapes) == expected
    assert all(a.text for a in overlay.layout.annotations)
    with pytest.raises(DataValidationError):
        pl.anomaly_overlay(production, anomalies, "nope")


def test_importance_bar() -> None:
    fig = pl.importance_bar(pd.Series({"a": 1.0, "b": 2.0}), title="t", x_title="x")
    assert fig.data[0].orientation == "h"
    assert fig.data[0].marker.color == pl.SERIES_COLORS[0]


def test_plotting_requires_viz_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(MissingDependencyError, match=r"wellcast\[viz\]"):
        pl.ipr_plot(ipr_curve("vogel", 2500.0, qmax=1000.0))


# =============================================================================
# End of module: tests/test_plotting.py
# =============================================================================
