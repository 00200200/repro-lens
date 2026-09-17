import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from repro_lens.cli import main
from repro_lens.comparison import compare_reports, read_report, render_comparison
from repro_lens.verify import verify


def recorded_report():
    run = {
        "returncode": 0,
        "metrics": {"score": 1},
        "runtime": {"library": "1.0"},
        "artifacts_sha256": {"predictions.json": "a" * 64},
    }
    return {
        "schema_version": 1,
        "kind": "repeatability_test",
        "status": "matched",
        "configuration": {
            "command": ["/missing/interpreter", "train.py", "{output}"],
            "inputs": ["train.py", "data.csv"],
            "metrics": ["score"],
            "artifacts": ["predictions.json"],
            "timeout": 60,
            "atol": 0,
            "rtol": 0,
            "result": "result.json",
        },
        "environment": {"runner_python": "3.11", "platform": "test", "machine": "test"},
        "inputs_sha256": {"train.py": "b" * 64, "data.csv": "c" * 64},
        "git": {"commit": None, "dirty": None},
        "runs": [copy.deepcopy(run), copy.deepcopy(run)],
        "differences": [],
    }


def save(path, report):
    path.write_text(json.dumps(report))
    return path


def compare_pair(tmp_path, before, after):
    return compare_reports(
        save(tmp_path / "before.json", before), save(tmp_path / "after.json", after)
    )


def test_comparison_needs_only_reports_and_hashes_the_exact_evidence(tmp_path):
    report = recorded_report()
    result = compare_pair(tmp_path, report, report)
    assert result["status"] == "matched"
    assert (
        result["before"]["sha256"]
        == hashlib.sha256((tmp_path / "before.json").read_bytes()).hexdigest()
    )
    assert result["input_changes"] == {"added": [], "removed": [], "modified": []}
    # The command's interpreter, training code and artifact paths do not exist.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["after.json", "before.json"]


def test_equal_outputs_still_expose_changed_added_and_removed_inputs(tmp_path):
    before = recorded_report()
    after = copy.deepcopy(before)
    after["inputs_sha256"] = {"train.py": "d" * 64, "replacement.csv": "e" * 64}
    result = compare_pair(tmp_path, before, after)
    assert result["status"] == "matched"
    assert result["input_changes"] == {
        "added": ["replacement.csv"],
        "removed": ["data.csv"],
        "modified": ["train.py"],
    }
    assert "Input removed: data.csv" in render_comparison(result, "text")


@pytest.mark.parametrize("first,second", [(1, 2), (2**53, 2**53 + 1), (10**400, 10**400 + 1)])
def test_changed_metrics_cannot_pass_just_because_each_revision_repeats(tmp_path, first, second):
    before, after = recorded_report(), recorded_report()
    for run in before["runs"]:
        run["metrics"]["score"] = first
    for run in after["runs"]:
        run["metrics"]["score"] = second
    result = compare_pair(tmp_path, before, after)
    assert result["status"] == "mismatch"
    assert len(result["differences"]) == 4
    assert all("Metric score" in difference for difference in result["differences"])


def test_artifact_changes_are_visible_even_when_metrics_match(tmp_path):
    before, after = recorded_report(), recorded_report()
    for run in after["runs"]:
        run["artifacts_sha256"]["predictions.json"] = "d" * 64
    result = compare_pair(tmp_path, before, after)
    assert result["status"] == "mismatch"
    assert "Artifact predictions.json" in result["differences"][0]


def test_comparison_checks_all_run_pairs_without_assuming_tolerance_is_transitive(tmp_path):
    before, after = recorded_report(), recorded_report()
    for report, values in ((before, (1, 2)), (after, (2, 3))):
        report["configuration"]["atol"] = 1
        for run, value in zip(report["runs"], values, strict=True):
            run["metrics"]["score"] = value
    result = compare_pair(tmp_path, before, after)
    assert result["status"] == "mismatch"
    assert result["differences"] == ["Before run 1, after run 2: Metric score: 1 != 3"]
    after["runs"][1]["metrics"]["score"] = 2
    assert compare_pair(tmp_path, before, after)["status"] == "matched"


@pytest.mark.parametrize(
    "field,value",
    [
        ("atol", 100),
        ("rtol", 1),
        ("command", ["another-command", "{output}"]),
        ("inputs", ["replacement.csv"]),
        ("timeout", 30),
        ("result", "different.json"),
    ],
)
def test_changed_contract_is_not_comparable(tmp_path, field, value):
    before, after = recorded_report(), recorded_report()
    after["configuration"][field] = value
    result = compare_pair(tmp_path, before, after)
    assert result["status"] == "not_comparable"
    assert result["policy_changes"][field] == {
        "before": before["configuration"][field],
        "after": value,
    }


@pytest.mark.parametrize(
    "field,run_field", [("metrics", "metrics"), ("artifacts", "artifacts_sha256")]
)
def test_dropping_a_checked_output_does_not_hide_a_change(tmp_path, field, run_field):
    before, after = recorded_report(), recorded_report()
    after["configuration"][field] = []
    for run in after["runs"]:
        run[run_field] = {}
    result = compare_pair(tmp_path, before, after)
    assert result["status"] == "not_comparable"
    assert field in result["policy_changes"]


@pytest.mark.parametrize("field", ["runner_python", "platform", "machine", "experiment_runtime"])
def test_changed_recorded_environment_requires_review(tmp_path, field):
    before, after = recorded_report(), recorded_report()
    if field == "experiment_runtime":
        for run in after["runs"]:
            run["runtime"]["library"] = "2.0"
    else:
        after["environment"][field] = "changed"
    result = compare_pair(tmp_path, before, after)
    assert result["status"] == "not_comparable"
    assert field in result["environment_changes"]


def test_boolean_runtime_metadata_is_not_equal_to_a_numeric_value(tmp_path):
    before, after = recorded_report(), recorded_report()
    for run in before["runs"]:
        run["runtime"] = {"deterministic": True}
    for run in after["runs"]:
        run["runtime"] = {"deterministic": 1}
    assert compare_pair(tmp_path, before, after)["status"] == "not_comparable"


def test_runtime_metadata_types_must_agree_within_a_matched_report(tmp_path):
    report = recorded_report()
    report["runs"][0]["runtime"] = {"deterministic": True}
    report["runs"][1]["runtime"] = {"deterministic": 1}
    with pytest.raises(ValueError, match="contradict"):
        read_report(save(tmp_path / "report.json", report))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("kind", "static_check"),
        ("status", "error"),
        ("status", "mismatch"),
        ("differences", ["Artifact changed"]),
        ("configuration", {}),
        ("environment", {}),
        ("inputs_sha256", {}),
        ("inputs_sha256", {"train.py": "not-a-hash"}),
        ("runs", []),
        ("runs", [{"returncode": 0}, {"returncode": 0}]),
    ],
)
def test_incomplete_or_failed_evidence_cannot_be_used_as_a_baseline(tmp_path, field, value):
    report = recorded_report()
    report[field] = value
    with pytest.raises(ValueError):
        read_report(save(tmp_path / "bad.json", report))


@pytest.mark.parametrize(
    "field,value",
    [
        ("returncode", 1),
        ("returncode", False),
        ("metrics", {}),
        ("metrics", {"score": True}),
        ("metrics", {"score": "1"}),
        ("metrics", {"score": 1, "undeclared": 2}),
        ("metrics", {"score": 2}),
        ("artifacts_sha256", {}),
        ("artifacts_sha256", {"predictions.json": "d" * 64}),
        ("runtime", {"library": "2.0"}),
    ],
)
def test_run_evidence_must_support_the_claimed_match(tmp_path, field, value):
    report = recorded_report()
    report["runs"][1][field] = value
    with pytest.raises(ValueError):
        read_report(save(tmp_path / "bad.json", report))


@pytest.mark.parametrize(
    "text, message",
    [
        ('{"x":1,"x":2}', "Duplicate"),
        ('{"x":NaN}', "Nonfinite"),
        ('{"x":1e999}', "Nonfinite"),
        ('{"x":1e-400}', "Underflowing"),
        ('{"x":-1e-400}', "Underflowing"),
        ("[]", "Expected a verification report object"),
        ("broken", "Expecting"),
    ],
)
def test_reports_use_strict_json(tmp_path, text, message):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_report(path)


def test_oversized_report_is_rejected_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_lens.comparison.REPORT_LIMIT", 10)
    with pytest.raises(ValueError, match="exceeds"):
        read_report(save(tmp_path / "report.json", recorded_report()))


def test_cli_and_skill_compare_the_same_evidence_and_report_errors(tmp_path, capsys):
    path = save(tmp_path / "report.json", recorded_report())
    assert main(["compare", str(path), str(path), "--format", "json"]) == 0
    expected = json.loads(capsys.readouterr().out)
    launcher = Path(__file__).resolve().parents[1] / "skills/reproducibility/scripts/run.py"
    result = subprocess.run(
        [sys.executable, str(launcher), "compare", str(path), str(path), "--format", "json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == expected
    after = recorded_report()
    for run in after["runs"]:
        run["metrics"]["score"] = 2
    next_path = save(tmp_path / "after.json", after)
    assert main(["compare", str(path), str(next_path), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "mismatch"
    after["configuration"]["atol"] = 100
    save(next_path, after)
    assert main(["compare", str(path), str(next_path), "--format", "json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "not_comparable"
    next_path.write_text("broken")
    assert main(["compare", str(path), str(next_path), "--format", "json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_actual_agent_edit_can_repeat_but_change_the_baseline(tmp_path):
    (tmp_path / "pyproject.toml").write_text("""
[tool.repro-lens.verify]
command = ["{python}", "train.py", "{output}"]
inputs = ["train.py", "pyproject.toml"]
metrics = ["score"]
""")
    source = """import json, sys
from pathlib import Path
Path(sys.argv[1], 'result.json').write_text(json.dumps({'metrics': {'score': SCORE}}))
"""
    script = tmp_path / "train.py"
    script.write_text(source.replace("SCORE", "1"))
    before = verify(tmp_path)
    script.write_text(source.replace("SCORE", "2"))
    after = verify(tmp_path)
    assert before["status"] == after["status"] == "matched"
    comparison = compare_reports(Path(before["report_path"]), Path(after["report_path"]))
    assert comparison["status"] == "mismatch"
    assert comparison["input_changes"]["modified"] == ["train.py"]
