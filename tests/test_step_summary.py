"""Offline tests for GitHub Actions job-summary markdown (no network or Docker)."""

import pytest

from repro_lens.cli import main
from repro_lens.project import check, render_step_summary, write_step_summary


def project(tmp_path):
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src/train.py").write_text(
        "from sklearn.tree import DecisionTreeClassifier\nDecisionTreeClassifier()\n"
    )
    return tmp_path


def test_step_summary_table_and_collapsible_finding_details(tmp_path):
    report = check(project(tmp_path))
    summary = render_step_summary(report, "service")
    assert summary.startswith("## Repro Lens\n")
    assert "| Severity | Code | Location | Message |" in summary
    assert "`service/src/train.py:2:1`" in summary
    assert "<details>" in summary
    assert "<summary>Finding details</summary>" in summary
    assert "</details>" in summary
    assert "R101" in summary


def test_step_summary_escapes_pipes_and_newlines_in_table_cells():
    report = {
        "kind": "static_check",
        "files_checked": 1,
        "assurance": "Static screening only.",
        "findings": [
            {
                "path": "a|b.py",
                "line": 1,
                "column": 1,
                "code": "S902",
                "severity": "error",
                "message": "broken\nsecond|line",
                "suggestion": "Fix | it.",
            }
        ],
        "suppressed": [],
    }
    summary = render_step_summary(report)
    assert "a\\|b.py:1:1" in summary
    assert "broken second\\|line" in summary
    assert "Fix \\| it." in summary
    assert "\nsecond|line" not in summary


def test_step_summary_clean_static_check():
    report = {
        "kind": "static_check",
        "files_checked": 2,
        "assurance": "Static screening only.",
        "findings": [],
        "suppressed": [],
    }
    summary = render_step_summary(report)
    assert "No findings from the enabled checks." in summary
    assert "<details>" not in summary


def test_verify_mismatch_uses_output_differences_details():
    report = {
        "kind": "repeatability_test",
        "status": "mismatch",
        "assurance": "Two runs only.",
        "report_path": "/tmp/report.json",
        "differences": ["Metric score: 1 != 2", "Artifact out.bin: SHA-256 differs"],
    }
    summary = render_step_summary(report)
    assert "## Repro Lens — `mismatch`" in summary
    assert "| Status | `mismatch` |" in summary
    assert "<summary>Output differences</summary>" in summary
    assert "1. Metric score: 1 != 2" in summary
    assert "2. Artifact out.bin: SHA-256 differs" in summary


def test_compare_not_comparable_explains_why():
    report = {
        "kind": "report_comparison",
        "status": "not_comparable",
        "assurance": "Recorded outputs only.",
        "before": {"path": "/tmp/before.json"},
        "after": {"path": "/tmp/after.json"},
        "policy_changes": {"atol": {"before": 0, "after": 1e-6}},
        "environment_changes": {},
        "input_changes": {"added": [], "removed": [], "modified": ["data.csv"]},
        "differences": [],
    }
    summary = render_step_summary(report)
    assert "<summary>Why reports are not comparable</summary>" in summary
    assert "policy_changes: `atol`" in summary
    assert "Input modified: `data.csv`" in summary


def test_matched_verify_has_no_collapsible_section():
    report = {
        "kind": "repeatability_test",
        "status": "matched",
        "assurance": "Two runs only.",
        "report_path": "/tmp/report.json",
        "differences": [],
    }
    summary = render_step_summary(report)
    assert "No output differences." in summary
    assert "<details>" not in summary


def test_verify_with_warnings_renders_warnings_section():
    report = {
        "kind": "repeatability_test",
        "status": "matched",
        "assurance": "Two runs only.",
        "report_path": "/tmp/report.json",
        "differences": [],
        "warnings": ["Multi-threading variables unpinned: OMP_NUM_THREADS."],
    }
    summary = render_step_summary(report)
    assert "<summary>Warnings</summary>" in summary
    assert "OMP_NUM_THREADS" in summary


def test_unsupported_kind_is_rejected():
    with pytest.raises(ValueError, match="Unsupported report kind"):
        render_step_summary({"kind": "other"})


def test_write_step_summary_appends_only_when_env_is_set(tmp_path, monkeypatch):
    report = {
        "kind": "static_check",
        "files_checked": 0,
        "assurance": "Static screening only.",
        "findings": [],
        "suppressed": [],
    }
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    write_step_summary(report)
    target = tmp_path / "summary.md"
    target.write_text("preface\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    write_step_summary(report)
    text = target.read_text()
    assert text.startswith("preface\n## Repro Lens\n")
    assert "No findings from the enabled checks." in text


def test_cli_check_writes_job_summary_from_github_env(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    summary = tmp_path / "job-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert main(["check", "--root", str(tmp_path), "--format", "github"]) == 1
    out = capsys.readouterr().out
    assert "::warning" in out
    written = summary.read_text()
    assert "| Severity | Code | Location | Message |" in written
    assert "<summary>Finding details</summary>" in written
