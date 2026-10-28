# =============================================================================
# wellcast: consistent plotly theme and chart helpers
# -----------------------------------------------------------------------------
# Module:      wellcast.plotting
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

"""Interactive charts for production engineering, built on plotly.

Every helper returns a :class:`plotly.graph_objects.Figure` styled with the
``"wellcast"`` template:

* Series colors follow the entity, never its rank. Fluids keep fixed colors
  and dash patterns everywhere (oil green solid, gas red dashed, water blue
  dotted), so identity never relies on color alone.
* Categorical series take colors from :data:`SERIES_COLORS` in fixed order;
  scatter-type charts color at most three wells and show the rest in one
  neutral color with the well name on hover.
* Quantities with different units go into separate stacked panels sharing
  the time axis, never onto a second y-axis.

Requires the ``viz`` extra: ``pip install 'wellcast[viz]'``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from wellcast._validation import require_columns, require_extra

if TYPE_CHECKING:
    from wellcast.dca import DeclineModel

__all__ = [
    "FLUID_STYLES",
    "SERIES_COLORS",
    "anomaly_overlay",
    "chan_plot",
    "decline_plot",
    "hall_plot",
    "importance_bar",
    "ipr_plot",
    "predicted_vs_actual",
    "prediction_interval_plot",
    "production_history",
    "register_theme",
]

# Categorical palette in fixed order, validated for color-vision deficiency
# separation on adjacent pairs against a light surface (#fcfcfb).
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_MUTED = "#52514e"
_GRID = "#e6e5e0"
_NEUTRAL = "#8a8984"
_CRITICAL = "#d03b3b"

# Fluid identity: industry color convention plus a dash pattern as the
# secondary encoding (green and red are close for protan viewers).
FLUID_STYLES: dict[str, dict[str, str]] = {
    "oil_rate": {"color": "#008300", "dash": "solid", "label": "oil (bbl/d)"},
    "gas_rate": {"color": "#e34948", "dash": "dash", "label": "gas (Mscf/d)"},
    "water_rate": {"color": "#2a78d6", "dash": "dot", "label": "water (bbl/d)"},
    "liquid_rate": {"color": "#6250d6", "dash": "dashdot", "label": "liquid (bbl/d)"},
    "water_inj_rate": {
        "color": "#1baf7a",
        "dash": "solid",
        "label": "water injection (bbl/d)",
    },
}
_PRESSURE_STYLES: dict[str, dict[str, str]] = {
    "bhp": {"color": "#2a78d6", "dash": "solid", "label": "BHP (psi)"},
    "whp": {"color": "#eb6834", "dash": "dash", "label": "WHP (psi)"},
}
_MAX_SCATTER_SERIES = 3
_WELL_HOVER = "actual %{x:.4g}<br>predicted %{y:.4g}<br>%{text}<extra></extra>"


def _go() -> Any:
    return cast(Any, require_extra("plotly.graph_objects", "viz"))


# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------


def register_theme() -> str:
    """Register the ``"wellcast"`` plotly template and return its name.

    Returns
    -------
    str
        ``"wellcast"``, usable as ``template="wellcast"`` in any plotly
        figure.

    Raises
    ------
    MissingDependencyError
        If plotly is not installed.

    Examples
    --------
    >>> register_theme()
    'wellcast'
    """
    go = _go()
    pio = cast(Any, require_extra("plotly.io", "viz"))
    axis = {
        "gridcolor": _GRID,
        "linecolor": _GRID,
        "zerolinecolor": _GRID,
        "tickcolor": _GRID,
        "title": {"font": {"color": _TEXT_MUTED}},
        "tickfont": {"color": _TEXT_MUTED},
    }
    pio.templates["wellcast"] = go.layout.Template(
        layout={
            "colorway": list(SERIES_COLORS),
            "font": {
                "family": "Inter, Helvetica, Arial, sans-serif",
                "size": 13,
                "color": _TEXT,
            },
            "paper_bgcolor": _SURFACE,
            "plot_bgcolor": _SURFACE,
            "xaxis": axis,
            "yaxis": axis,
            "hovermode": "x unified",
            "hoverlabel": {"font": {"color": _TEXT}, "bgcolor": "white"},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
            "margin": {"l": 64, "r": 24, "t": 72, "b": 48},
            "title": {"x": 0.0, "xanchor": "left", "font": {"size": 16}},
        },
        data={"scatter": [go.Scatter(line={"width": 2}, marker={"size": 8})]},
    )
    return "wellcast"


def _apply_theme(fig: Any) -> Any:
    # Surfaces are also set on the figure itself: some renderers (Streamlit
    # among them) replace template-only background colors with their own.
    fig.update_layout(
        template=register_theme(), paper_bgcolor=_SURFACE, plot_bgcolor=_SURFACE
    )
    return fig


def _figure(**layout: Any) -> Any:
    go = _go()
    fig = _apply_theme(go.Figure())
    fig.update_layout(**layout)
    return fig


def _select_well(
    df: pd.DataFrame, well: str | None, by: str
) -> tuple[pd.DataFrame, str]:
    if by in df.columns:
        names = sorted(df[by].astype(str).unique())
        chosen = well if well is not None else names[0]
        subset: pd.DataFrame = df[df[by].astype(str) == chosen]
        return subset, chosen
    return df, well or ""


# -----------------------------------------------------------------------------
# Production history
# -----------------------------------------------------------------------------


def production_history(
    df: pd.DataFrame,
    well: str | None = None,
    *,
    rates: Sequence[str] = ("oil_rate", "gas_rate", "water_rate"),
    pressures: Sequence[str] = ("bhp", "whp"),
    by: str = "well",
) -> Any:
    """Rates and pressures of one well in two stacked panels.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table with a ``DatetimeIndex``.
    well : str, optional
        Well to show. Default: the first well, alphabetically, with data in
        the requested columns.
    rates, pressures : sequence of str, optional
        Columns for the upper (rates) and lower (pressures) panel; absent
        columns are skipped.
    by : str, optional
        Well column. Default ``"well"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with a shared time axis.

    Examples
    --------
    >>> from wellcast.datasets import make_synthetic_field
    >>> field = make_synthetic_field(n_wells=1, years=1, n_injectors=0)
    >>> fig = production_history(field.production)
    >>> [t.name for t in fig.data]
    ['oil (bbl/d)', 'gas (Mscf/d)', 'water (bbl/d)', 'BHP (psi)', 'WHP (psi)']
    """
    go = _go()
    subplots = cast(Any, require_extra("plotly.subplots", "viz"))
    if well is None and by in df.columns:
        # Default to the first well that has data to show (an injector has no
        # oil rate, for example).
        shown = [c for c in (*rates, *pressures) if c in df.columns]
        with_data = df.loc[df[shown].notna().any(axis=1), by] if shown else df[by]
        names = sorted(with_data.astype(str).unique())
        well = names[0] if names else None
    data, name = _select_well(df, well, by)
    rate_cols = [c for c in rates if c in data.columns and data[c].notna().any()]
    pressure_cols = [
        c for c in pressures if c in data.columns and data[c].notna().any()
    ]
    panels = [cols for cols in (rate_cols, pressure_cols) if cols]
    fig = subplots.make_subplots(
        rows=max(len(panels), 1),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )
    for row, cols in enumerate(panels, start=1):
        unit = "rate" if cols is rate_cols else "pressure (psi)"
        fig.update_yaxes(title_text=unit, row=row, col=1)
        for column in cols:
            style = FLUID_STYLES.get(column) or _PRESSURE_STYLES.get(
                column, {"color": SERIES_COLORS[0], "dash": "solid", "label": column}
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=style["label"],
                    line={"color": style["color"], "dash": style["dash"], "width": 2},
                ),
                row=row,
                col=1,
            )
    _apply_theme(fig)
    fig.update_layout(
        title=f"Production history {name}".strip(),
        height=260 * max(len(panels), 1) + 120,
    )
    return fig


# -----------------------------------------------------------------------------
# Decline curves
# -----------------------------------------------------------------------------


def decline_plot(
    model: DeclineModel,
    rates: pd.Series | None = None,
    *,
    forecast_days: int = 3650,
    econ_limit: float | None = None,
    title: str | None = None,
) -> Any:
    """Observed rates, fitted Arps curve and bootstrap P90 to P10 band.

    Parameters
    ----------
    model : DeclineModel
        A fitted model (bands are drawn when it was bootstrapped).
    rates : pandas.Series, optional
        Observed rates to overlay.
    forecast_days : int, optional
        Horizon after the last observation, days. Default 3650.
    econ_limit : float, optional
        Economic limit rate. The forecast stops when it is reached and the
        limit is drawn as a horizontal line.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with a logarithmic rate axis.
    """
    go = _go()
    t_end = model.t_last_ + forecast_days
    if econ_limit is not None:
        t_end = min(t_end, max(model.time_to_limit(econ_limit), model.t_last_))
    t = np.linspace(0.0, t_end, 400)
    x: Any = (
        model.start_ + pd.to_timedelta(t, unit="D") if model.start_ is not None else t
    )
    # Keep the useful range visible: the axis floor sits just below the lowest
    # observed, forecast or economic-limit rate.
    floors = [float(np.min(np.asarray(model.predict(t))))]
    ceiling = model.qi_
    if rates is not None and (rates > 0).any():
        floors.append(float(rates[rates > 0].min()))
        ceiling = max(ceiling, float(rates.max()))
    if econ_limit is not None:
        floors.append(econ_limit)
    y_range = [np.log10(max(min(floors), 1e-9) / 1.5), np.log10(ceiling * 1.5)]
    fig = _figure(
        title=title or f"Arps {model.model_} decline (b = {model.b_:.2f})",
        yaxis={"type": "log", "title": "rate", "range": y_range},
        xaxis={"title": "date" if model.start_ is not None else "days"},
    )
    color = SERIES_COLORS[0]
    if model._samples is not None:
        bands = model.predict_interval(t)
        fig.add_trace(
            go.Scatter(
                x=x, y=bands["p10"], name="P10", line={"width": 0}, showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bands["p90"],
                name="P90 to P10",
                fill="tonexty",
                line={"width": 0},
                fillcolor="rgba(42, 120, 214, 0.18)",
            )
        )
    if rates is not None:
        fig.add_trace(
            go.Scatter(
                x=rates.index,
                y=rates,
                mode="markers",
                name="observed",
                marker={"size": 5, "color": _NEUTRAL, "opacity": 0.6},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x, y=model.predict(t), name=f"{model.model_} fit", line={"color": color}
        )
    )
    if econ_limit is not None:
        fig.add_hline(
            y=econ_limit,
            line={"color": _TEXT_MUTED, "dash": "dot", "width": 1},
            annotation_text=f"economic limit {econ_limit:g}",
            annotation_font_color=_TEXT_MUTED,
        )
    return fig


def importance_bar(importance: pd.Series, *, title: str, x_title: str) -> Any:
    """Horizontal bar chart of feature importances, largest on top.

    Parameters
    ----------
    importance : pandas.Series
        Importance per feature, sorted ascending (largest last).
    title : str
        Figure title.
    x_title : str
        Axis title for the importance values.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart in the wellcast theme.

    Examples
    --------
    >>> import pandas as pd
    >>> fig = importance_bar(
    ...     pd.Series({"whp": 1.0, "choke": 3.0}), title="t", x_title="x"
    ... )
    >>> list(fig.data[0].y)
    ['whp', 'choke']
    """
    go = _go()
    fig = _figure(
        title=title,
        xaxis={"title": x_title},
        yaxis={"automargin": True},
        hovermode="closest",
        height=max(320, 28 * len(importance) + 140),
    )
    fig.add_trace(
        go.Bar(
            x=importance.to_numpy(),
            y=[str(i) for i in importance.index],
            orientation="h",
            marker={"color": SERIES_COLORS[0], "cornerradius": 4},
            name="importance",
            showlegend=False,
        )
    )
    return fig


# -----------------------------------------------------------------------------
# Inflow performance
# -----------------------------------------------------------------------------


def ipr_plot(
    curves: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    operating_point: tuple[float, float] | None = None,
    title: str = "Inflow performance",
) -> Any:
    """Plot one or more IPR curves (pressure against rate).

    Parameters
    ----------
    curves : pandas.DataFrame or mapping of str to DataFrame
        Output of :func:`wellcast.ipr.ipr_curve`, or several keyed by label.
    operating_point : tuple of float, optional
        ``(rate, pwf)`` of a measured test, drawn as a marker.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Rate on x, flowing bottomhole pressure on y.

    Examples
    --------
    >>> from wellcast.ipr import ipr_curve
    >>> fig = ipr_plot({"Vogel": ipr_curve("vogel", 2500.0, qmax=1000.0)})
    >>> fig.data[0].name
    'Vogel'
    """
    go = _go()
    named = {"IPR": curves} if isinstance(curves, pd.DataFrame) else dict(curves)
    fig = _figure(
        title=title,
        xaxis={"title": "rate (STB/d)", "rangemode": "tozero"},
        yaxis={"title": "flowing bottomhole pressure (psi)", "rangemode": "tozero"},
        hovermode="closest",
    )
    for i, (label, curve) in enumerate(named.items()):
        require_columns(curve, ["pwf", "rate"])
        fig.add_trace(
            go.Scatter(
                x=curve["rate"],
                y=curve["pwf"],
                name=label,
                line={"color": SERIES_COLORS[i % len(SERIES_COLORS)]},
            )
        )
    if operating_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[operating_point[0]],
                y=[operating_point[1]],
                mode="markers",
                name="test point",
                marker={
                    "size": 10,
                    "color": _TEXT,
                    "line": {"color": _SURFACE, "width": 2},
                },
            )
        )
    return fig


# -----------------------------------------------------------------------------
# Machine learning
# -----------------------------------------------------------------------------


def predicted_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    wells: pd.Series | None = None,
    *,
    title: str = "Predicted vs actual",
) -> Any:
    """Cross-plot of predictions against observations with a 1:1 line.

    Parameters
    ----------
    y_true, y_pred : pandas.Series
        Observed and predicted values, aligned by position.
    wells : pandas.Series, optional
        Well names. Up to three wells are colored; with more, all points
        share one color and the well name appears on hover.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Square-aspect cross-plot.
    """
    go = _go()
    yt = y_true.to_numpy(dtype=np.float64, na_value=np.nan)
    yp = y_pred.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(yt) & np.isfinite(yp)
    low = float(np.min(np.r_[yt[finite], yp[finite]])) if finite.any() else 0.0
    high = float(np.max(np.r_[yt[finite], yp[finite]])) if finite.any() else 1.0
    fig = _figure(
        title=title,
        xaxis={"title": "actual"},
        yaxis={"title": "predicted", "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest",
    )
    names = None if wells is None else wells.astype(str).to_numpy()
    groups = [] if names is None else sorted(pd.unique(names))
    marker = {"size": 6, "opacity": 0.5}
    if names is not None and len(groups) <= _MAX_SCATTER_SERIES:
        for i, well in enumerate(groups):
            mask = names == well
            fig.add_trace(
                go.Scatter(
                    x=yt[mask],
                    y=yp[mask],
                    mode="markers",
                    name=well,
                    marker={**marker, "color": SERIES_COLORS[i]},
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=yt,
                y=yp,
                mode="markers",
                name="observations",
                marker={**marker, "color": SERIES_COLORS[0]},
                text=names,
                hovertemplate=_WELL_HOVER if names is not None else None,
            )
        )
    # The reference line goes on top so dense point clouds do not hide it.
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="1:1",
            line={"color": _TEXT, "dash": "dash", "width": 1.5},
        )
    )
    return fig


def prediction_interval_plot(
    actual: pd.Series,
    interval: pd.DataFrame,
    *,
    title: str = "Prediction with interval",
) -> Any:
    """Observed values against predictions with a shaded interval.

    Parameters
    ----------
    actual : pandas.Series
        Observed values, indexed like ``interval`` (typically by date).
    interval : pandas.DataFrame
        Output of :meth:`wellcast.ml.Predictor.predict_interval`, with
        columns ``lower``, ``prediction`` and ``upper``.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Time-series figure with the interval as a band.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=3, freq="D")
    >>> band = pd.DataFrame(
    ...     {
    ...         "lower": [1.0, 2.0, 3.0],
    ...         "prediction": [2.0, 3.0, 4.0],
    ...         "upper": [3.0, 4.0, 5.0],
    ...     },
    ...     index=idx,
    ... )
    >>> fig = prediction_interval_plot(pd.Series([2.1, 2.9, 4.2], index=idx), band)
    >>> [t.name for t in fig.data]
    ['upper', 'interval', 'prediction', 'actual']
    """
    go = _go()
    require_columns(interval, ["lower", "prediction", "upper"])
    fig = _figure(title=title)
    x = interval.index
    fig.add_trace(
        go.Scatter(
            x=x, y=interval["upper"], name="upper", line={"width": 0}, showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=interval["lower"],
            name="interval",
            fill="tonexty",
            line={"width": 0},
            fillcolor="rgba(42, 120, 214, 0.18)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=interval["prediction"],
            name="prediction",
            line={"color": SERIES_COLORS[0]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            name="actual",
            line={"color": SERIES_COLORS[1], "dash": "dot"},
        )
    )
    return fig


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def chan_plot(chan: pd.DataFrame, well: str | None = None) -> Any:
    """Chan diagnostic plot: WOR and WOR' against time on log-log axes.

    Parameters
    ----------
    chan : pandas.DataFrame
        Output of :func:`wellcast.diagnostics.chan_plot_data`.
    well : str, optional
        Well to show. Default: the first well.

    Returns
    -------
    plotly.graph_objects.Figure
        Log-log figure with two series.
    """
    go = _go()
    data, name = _select_well(chan, well, "well")
    positive = data[data["wor_derivative"] > 0]
    fig = _figure(
        title=f"Chan plot {name}".strip(),
        xaxis={"type": "log", "title": "time (days)"},
        yaxis={"type": "log", "title": "WOR, WOR' (1/day)"},
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"], y=data["wor"], name="WOR", line={"color": SERIES_COLORS[0]}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=positive["t"],
            y=positive["wor_derivative"],
            name="WOR'",
            line={"color": SERIES_COLORS[1], "dash": "dash"},
        )
    )
    return fig


def hall_plot(
    hall: pd.DataFrame, change: pd.Series | None = None, *, title: str = "Hall plot"
) -> Any:
    """Hall plot of the pressure-time integral against cumulative injection.

    Parameters
    ----------
    hall : pandas.DataFrame
        Output of :func:`wellcast.diagnostics.hall_plot_data`.
    change : pandas.Series, optional
        Output of :func:`wellcast.diagnostics.hall_slope_change`; the
        breakpoint is marked and both slopes are labelled.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with cumulative injection on x.
    """
    go = _go()
    fig = _figure(
        title=title,
        xaxis={"title": "cumulative injection (bbl)"},
        yaxis={"title": "Hall integral (psi day)"},
        hovermode="closest",
    )
    fig.add_trace(
        go.Scatter(
            x=hall["cum_injection"],
            y=hall["hall_integral"],
            name="Hall integral",
            line={"color": SERIES_COLORS[0]},
        )
    )
    if change is not None:
        x_break = float(hall.loc[change["break_date"], "cum_injection"])
        fig.add_vline(
            x=x_break,
            line={"color": _TEXT_MUTED, "dash": "dash", "width": 1},
            annotation_text=(
                f"slope {change['slope_before']:.3g} to {change['slope_after']:.3g} "
                f"(x{change['ratio']:.2f})"
            ),
            annotation_font_color=_TEXT_MUTED,
        )
    return fig


def anomaly_overlay(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    column: str,
    well: str | None = None,
    *,
    by: str = "well",
) -> Any:
    """Time series of one signal with flagged periods shaded.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table with a ``DatetimeIndex``.
    anomalies : pandas.DataFrame
        Output of :func:`wellcast.diagnostics.detect_anomalies`.
    column : str
        Signal to draw.
    well : str, optional
        Well to show. Default: the first well.
    by : str, optional
        Well column. Default ``"well"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Line chart with one labelled rectangle per anomaly period.
    """
    go = _go()
    require_columns(df, [column])
    data, name = _select_well(df, well, by)
    periods = anomalies
    if by in anomalies.columns and name:
        periods = anomalies[anomalies[by] == name]
    style = FLUID_STYLES.get(column) or _PRESSURE_STYLES.get(
        column, {"color": SERIES_COLORS[0], "dash": "solid", "label": column}
    )
    fig = _figure(title=f"{style['label']} with anomalies {name}".strip())
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            name=style["label"],
            line={"color": style["color"], "dash": style["dash"]},
        )
    )
    # Single-day periods would be invisible on a multi-year axis, so every band
    # is at least 0.5 percent of the time span wide.
    span = data.index.max() - data.index.min() if len(data) else pd.Timedelta(days=1)
    half_width = max(pd.Timedelta(hours=12), span * 0.0025)
    for _, period in periods.iterrows():
        # Status color plus a text label, so the flag never relies on color.
        fig.add_vrect(
            x0=period["start"] - half_width,
            x1=period["end"] + half_width,
            fillcolor=_CRITICAL,
            opacity=0.25,
            line_width=0,
            annotation_text=str(period["detector"]),
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=_TEXT_MUTED,
        )
    return fig


# =============================================================================
# End of module: wellcast.plotting
# =============================================================================
