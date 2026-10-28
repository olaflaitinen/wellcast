# =============================================================================
# wellcast: MkDocs build hooks for the documentation site
# -----------------------------------------------------------------------------
# Module:      scripts/mkdocs_hooks.py
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

"""MkDocs hooks used by ``mkdocs.yml``.

The tutorial notebooks are executed during the build. Plotly's default output
in a headless kernel is its own MIME type, which the HTML export drops, so the
build switches plotly to the ``notebook_connected`` renderer (HTML that loads
plotly.js from a CDN). The kernels inherit the environment variable; running
the notebooks in Jupyter, VS Code or Colab is unaffected.
"""

from __future__ import annotations

import os
from typing import Any


def on_config(config: Any) -> Any:
    """Select the plotly renderer before notebooks are executed."""
    os.environ["PLOTLY_RENDERER"] = "notebook_connected"
    return config


# =============================================================================
# End of module: scripts/mkdocs_hooks.py
# =============================================================================
