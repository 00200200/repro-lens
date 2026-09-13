#!/usr/bin/env python3
"""Run the bundled stdlib CLI directly from a source/plugin checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repro_lens.cli import main  # noqa: E402

raise SystemExit(main())
