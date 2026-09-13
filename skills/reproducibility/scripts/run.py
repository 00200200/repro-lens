#!/usr/bin/env python3
"""Use the installed engine, or the same engine in the complete plugin checkout."""

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("repro_lens") is None:
    bundled = Path(__file__).resolve().parents[3] / "src"
    if not (bundled / "repro_lens" / "cli.py").is_file():
        raise SystemExit(
            "Install Repro Lens from its reviewed source checkout, or use the complete plugin."
        )
    sys.path.insert(0, str(bundled))

from repro_lens.cli import main  # noqa: E402

raise SystemExit(main())
