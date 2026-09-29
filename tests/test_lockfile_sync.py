"""Offline tests for P204 lockfile/environment synchronicity."""

from pathlib import Path

from repro_lens.analysis import RULES
from repro_lens.project import (
    check,
    lockfile_environment_mismatches,
    lockfile_sync_mismatches,
    normalize_package_name,
    read_lockfile_versions,
)
from repro_lens.verify import verify


def test_normalize_package_name_follows_pep503():
    assert normalize_package_name("PyYAML") == "pyyaml"
    assert normalize_package_name("curious_package.name") == "curious-package-name"


def test_lockfile_sync_mismatches_ignore_one_sided_packages():
    locked = {"numpy": {"2.0.0"}, "only-locked": {"1.0.0"}}
    installed = {"numpy": "2.0.0", "only-installed": "9.9.9"}
    assert lockfile_sync_mismatches(locked, installed) == []


def test_lockfile_sync_mismatches_report_version_drift():
    locked = {"numpy": {"2.0.0", "2.1.0"}, "pytest": {"8.0.0"}}
    installed = {"numpy": "1.26.4", "pytest": "8.0.0"}
    assert lockfile_sync_mismatches(locked, installed) == [
        "numpy is 1.26.4 but lockfile has 2.0.0, 2.1.0"
    ]


def test_read_uv_and_poetry_lockfiles(tmp_path):
    uv = tmp_path / "uv.lock"
    uv.write_text(
        'version = 1\n\n[[package]]\nname = "NumPy"\nversion = "2.2.6"\n'
        '[[package]]\nname = "scipy"\nversion = "1.17.1"\n'
        '[[package]]\nname = "scipy"\nversion = "1.18.1"\n'
    )
    poetry = tmp_path / "poetry.lock"
    poetry.write_text('[[package]]\nname = "Requests"\nversion = "2.31.0"\n')
    assert read_lockfile_versions(uv) == {
        "numpy": {"2.2.6"},
        "scipy": {"1.17.1", "1.18.1"},
    }
    assert read_lockfile_versions(poetry) == {"requests": {"2.31.0"}}


def test_lockfile_environment_mismatches_prefer_uv_lock(tmp_path):
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "demo"\nversion = "1.0.0"\n')
    (tmp_path / "poetry.lock").write_text('[[package]]\nname = "demo"\nversion = "9.0.0"\n')
    name, mismatches = lockfile_environment_mismatches(tmp_path, installed={"demo": "9.0.0"})
    assert name == "uv.lock"
    assert mismatches == ["demo is 9.0.0 but lockfile has 1.0.0"]


def test_no_lockfile_skips_sync_check(tmp_path):
    assert lockfile_environment_mismatches(tmp_path, installed={"numpy": "1"}) == (None, [])


def _verify_project(tmp_path: Path, lock_body: str | None = None):
    (tmp_path / "train.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1], 'result.json').write_text("
        "json.dumps({'metrics': {'score': 1}}))\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[tool.repro-lens.verify]\n"
        'command = ["{python}", "train.py", "{output}"]\n'
        'inputs = ["train.py", "pyproject.toml"]\n'
        'metrics = ["score"]\n'
    )
    if lock_body is not None:
        (tmp_path / "uv.lock").write_text(lock_body)


def test_verify_stops_when_lockfile_disagrees(tmp_path):
    _verify_project(
        tmp_path,
        '[[package]]\nname = "pytest"\nversion = "0.0.0"\n',
    )
    result = verify(tmp_path)
    assert result["status"] == "error"
    assert "P204" in result["error"]
    assert "pytest" in result["error"]
    assert result["runs"] == []


def test_verify_allows_matching_or_unrelated_lock_entries(tmp_path):
    _verify_project(
        tmp_path,
        '[[package]]\nname = "not-installed-anywhere-xyz"\nversion = "1.2.3"\n',
    )
    result = verify(tmp_path)
    assert result["status"] == "matched"


def test_check_reports_p204_when_verify_configured(tmp_path):
    _verify_project(
        tmp_path,
        '[[package]]\nname = "pytest"\nversion = "0.0.0"\n',
    )
    report = check(tmp_path)
    assert "P204" in RULES
    assert [f["code"] for f in report["findings"] if f["code"] == "P204"]
    assert any("pytest" in f["message"] for f in report["findings"] if f["code"] == "P204")


def test_check_without_verify_policy_ignores_lockfile_drift(tmp_path):
    (tmp_path / "train.py").write_text("print(1)\n")
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "pytest"\nversion = "0.0.0"\n')
    assert check(tmp_path)["findings"] == []
