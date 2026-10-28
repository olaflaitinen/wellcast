# =============================================================================
# wellcast: Streamlit app entry point with multipage navigation
# -----------------------------------------------------------------------------
# Module:      app/streamlit_app.py
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

"""Entry point of the wellcast Streamlit app (``wellcast app``)."""

from __future__ import annotations

import streamlit as st

import common

icon = common.asset("icon.svg")
st.set_page_config(
    page_title="wellcast",
    page_icon=str(icon) if icon else None,
    layout="wide",
)
logo = common.asset("logo.svg")
if logo is not None:
    st.logo(str(logo), icon_image=str(icon) if icon else None, size="large")

pages = [
    st.Page("pages/data.py", title="Data", icon=":material/upload_file:", default=True),
    st.Page("pages/explore.py", title="Explore", icon=":material/monitoring:"),
    st.Page("pages/predict.py", title="Predict", icon=":material/model_training:"),
    st.Page("pages/decline.py", title="Decline", icon=":material/trending_down:"),
    st.Page("pages/diagnose.py", title="Diagnose", icon=":material/troubleshoot:"),
    st.Page("pages/report.py", title="Report", icon=":material/description:"),
]
navigation = st.navigation(pages)
common.sidebar()
navigation.run()

# =============================================================================
# End of module: app/streamlit_app.py
# =============================================================================
