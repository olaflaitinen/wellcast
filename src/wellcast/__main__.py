# =============================================================================
# wellcast: entry point for ``python -m wellcast``
# -----------------------------------------------------------------------------
# Module:      wellcast.__main__
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

"""Run the wellcast command-line interface with ``python -m wellcast``."""

import sys

from wellcast.cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    sys.exit(main())

# =============================================================================
# End of module: wellcast.__main__
# =============================================================================
