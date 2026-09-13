"""Run the live checker's JavaScript state tests when Node.js is available."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_live_checker_recovers_from_worker_failures():
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests/checker_state.test.mjs")],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
