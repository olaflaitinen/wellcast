# =============================================================================
# wellcast: end-to-end tests of the Streamlit app with AppTest
# -----------------------------------------------------------------------------
# Module:      tests/test_app.py
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

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
PAGES = ("explore", "decline", "diagnose", "predict", "report")


def _app_with_demo() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    app.button(key="load_demo_page").click().run()
    return app


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_start_page_offers_demo_data() -> None:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    _assert_clean(app)
    assert app.title[0].value == "Data"
    assert any(b.label == "Load demo data" for b in app.button)


def test_demo_data_and_quality_control() -> None:
    app = _app_with_demo()
    _assert_clean(app)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Wells"] == "6"
    assert metrics["Impossible values"] == "0"
    # Structurally empty columns (oil rate of the injector) are hidden.
    report = app.dataframe[0].value
    assert (report["missing"] < report["n"]).all()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_demo_data(page: str) -> None:
    app = _app_with_demo()
    app.switch_page(f"pages/{page}.py").run()
    _assert_clean(app)


def test_pages_without_data_show_the_demo_prompt() -> None:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.switch_page("pages/decline.py").run()
    _assert_clean(app)
    assert any("No data loaded" in i.value for i in app.info)


def test_predict_trains_and_reports_metrics() -> None:
    app = _app_with_demo()
    app.switch_page("pages/predict.py").run()
    next(b for b in app.button if b.label == "Train and validate").click().run()
    _assert_clean(app)
    labels = [m.label for m in app.metric]
    assert {"MAE", "RMSE", "MAPE", "R2"} <= set(labels)


def test_report_builds_html() -> None:
    app = _app_with_demo()
    app.switch_page("pages/report.py").run()
    next(b for b in app.button if b.label == "Build report").click().run()
    _assert_clean(app)
    html = app.session_state["report_html"]
    assert html.startswith("<!doctype html>")
    assert "Decline and EUR (oil)" in html
    assert "wellcast" in html


# =============================================================================
# End of module: tests/test_app.py
# =============================================================================
