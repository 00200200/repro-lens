import json
import textwrap
from pathlib import Path

import pytest

from repro_lens.cli import main
from repro_lens.verify import read_verify_config, verify


def experiment(tmp_path, body, extra="", metrics='["score"]', artifacts="[]"):
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
        import json, os, sys, time
        from pathlib import Path
        out = Path(sys.argv[1])
    """)
        + textwrap.dedent(body)
    )
    (tmp_path / "pyproject.toml").write_text(f"""
[tool.repro-lens.verify]
command = ["{{python}}", "train.py", "{{output}}"]
inputs = ["train.py", "pyproject.toml"]
metrics = {metrics}
artifacts = {artifacts}
{extra}
""")


def test_two_fresh_processes_match_and_leave_evidence(tmp_path):
    experiment(
        tmp_path,
        """
    (out / 'result.json').write_text(json.dumps({'metrics': {'score': 0.9}}))
    (out / 'pid.txt').write_text(str(os.getpid()))
    """,
    )
    result = verify(tmp_path)
    assert result["status"] == "matched"
    dirs = [Path(run["output"]) for run in result["runs"]]
    assert dirs[0] != dirs[1]
    assert (dirs[0] / "pid.txt").read_text() != (dirs[1] / "pid.txt").read_text()
    saved = json.loads(Path(result["report_path"]).read_text())
    assert saved["inputs_sha256"].keys() == {"train.py", "pyproject.toml"}


def test_nondeterminism_is_detected(tmp_path):
    experiment(
        tmp_path,
        "(out / 'result.json').write_text(json.dumps({'metrics': {'score': os.getpid()}}))",
    )
    result = verify(tmp_path)
    assert result["status"] == "mismatch"
    assert "score" in result["differences"][0]


@pytest.mark.parametrize(
    "first, second, extra, expected",
    [
        (2**53, 2**53 + 1, "", "mismatch"),
        (-(2**53), -(2**53 + 1), "", "mismatch"),
        (2**53 + 1, float(2**53), "", "mismatch"),
        (10**400, 10**400, "", "matched"),
        (10**400, 10**400 + 1, "", "mismatch"),
        (10**400, 10**400 + 1, "atol=1", "matched"),
        (10**400, 10**400 + 2, "atol=1", "mismatch"),
        (10.0, 10.125, "atol=0.125", "matched"),
        (10.0, 10.25, "atol=0.125", "mismatch"),
        (100, 101, "rtol=0.01", "matched"),
        (101, 100, "rtol=0.01", "matched"),
        (100, 102, "rtol=0.01", "mismatch"),
        (10**400, 2 * 10**400, "rtol=0.5", "matched"),
        (10**400, 3 * 10**400, "rtol=0.5", "mismatch"),
        (1.7e308, -1.7e308, "rtol=1.1", "mismatch"),
        (0.0, 5e-324, "rtol=0.75", "mismatch"),
        (1, 1.0, "", "matched"),
        (0, -0.0, "", "matched"),
    ],
    ids=[
        "adjacent-large-integers",
        "negative-large-integers",
        "mixed-integer-and-float",
        "equal-beyond-float-range",
        "different-beyond-float-range",
        "absolute-tolerance-boundary-large",
        "absolute-tolerance-exceeded-large",
        "absolute-tolerance-boundary-float",
        "absolute-tolerance-exceeded-float",
        "relative-tolerance",
        "relative-tolerance-symmetric",
        "relative-tolerance-exceeded",
        "relative-tolerance-boundary-large",
        "relative-tolerance-exceeded-large",
        "float-arithmetic-overflow",
        "float-arithmetic-underflow",
        "equal-mixed-types",
        "signed-zero",
    ],
)
def test_metric_comparison_preserves_precision(tmp_path, first, second, extra, expected):
    experiment(
        tmp_path,
        f"""
    score = {first!r} if out.name == 'run-1' else {second!r}
    (out / 'result.json').write_text(json.dumps({{'metrics': {{'score': score}}}}))
    """,
        extra=extra,
    )
    result = verify(tmp_path)
    assert result["status"] == expected
    assert [run["metrics"]["score"] for run in result["runs"]] == [first, second]
    saved = json.loads(Path(result["report_path"]).read_text())
    assert saved["status"] == expected
    assert [run["metrics"]["score"] for run in saved["runs"]] == [first, second]


def test_artifact_mismatch_even_when_metric_matches(tmp_path):
    experiment(
        tmp_path,
        """
    (out / 'result.json').write_text(json.dumps({'metrics': {'score': 1.0}}))
    (out / 'predictions.txt').write_text(str(os.getpid()))
    """,
        artifacts='["predictions.txt"]',
    )
    result = verify(tmp_path)
    assert result["status"] == "mismatch"
    assert "Artifact" in result["differences"][0]


@pytest.mark.parametrize(
    "body, reason",
    [
        ("sys.exit(3)", "exited with 3"),
        ("pass", "Missing"),
        ("(out / 'result.json').write_text('{\"metrics\": {}}')", "Missing"),
        ("(out / 'result.json').write_text('{\"metrics\": {\"score\": NaN}}')", "Nonfinite"),
        ("(out / 'result.json').write_text('{\"metrics\": {\"score\": 1e999}}')", "Nonfinite"),
        ("(out / 'result.json').write_text('{\"metrics\": {\"score\": true}}')", "nonnumeric"),
        ("(out / 'result.json').write_text('bad json')", "Expecting"),
    ],
)
def test_failed_or_invalid_results_never_pass(tmp_path, body, reason):
    experiment(tmp_path, body)
    result = verify(tmp_path)
    assert result["status"] == "error"
    assert reason in result["error"]


@pytest.mark.parametrize(
    "payload, reason",
    [
        ('{"metrics":{"score":1},"runtime":{"elapsed":1e999}}', "Nonfinite JSON value"),
        ('{"metrics":{"score":1},"runtime":{"elapsed":-1e999}}', "Nonfinite JSON value"),
        ('{"metrics":{"score":1},"runtime":{"samples":[1e999]}}', "Nonfinite JSON value"),
        ('{"metrics":{"score":1,"unused":1e999}}', "Nonfinite JSON value"),
        ('{"metrics":{"score":1},"runtime":{"elapsed":NaN}}', "Nonfinite JSON value"),
        ('{"metrics":{"score":0.1,"score":0.9}}', "Duplicate JSON key 'score'"),
        ('{"metrics":{"score":0.9},"metrics":{"score":1}}', "Duplicate JSON key 'metrics'"),
        (r'{"metrics":{"score":0.1,"\u0073core":0.9}}', "Duplicate JSON key 'score'"),
        ('{"metrics":{"score":1},"runtime":{"env":"a","env":"b"}}', "Duplicate JSON key 'env'"),
    ],
    ids=[
        "overflowing-runtime",
        "negative-overflow",
        "overflow-in-array",
        "undeclared-overflow",
        "runtime-nan",
        "duplicate-metric",
        "duplicate-metrics-object",
        "escaped-duplicate-key",
        "duplicate-runtime-key",
    ],
)
@pytest.mark.parametrize("invalid_run", [1, 2], ids=["first-run", "second-run"])
def test_invalid_json_retains_evidence(tmp_path, capsys, payload, reason, invalid_run):
    valid_payload = '{"metrics":{"score":1}}'
    experiment(
        tmp_path,
        f"payload = {payload!r} if out.name == 'run-{invalid_run}' else {valid_payload!r}\n"
        "print('experiment finished')\n(out / 'result.json').write_text(payload)",
    )
    assert main(["verify", "--root", str(tmp_path), "--format", "json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    saved = json.loads(Path(report["report_path"]).read_text())
    assert saved == report
    assert reason in report["error"]
    assert len(report["runs"]) == invalid_run
    output = Path(report["runs"][-1]["output"])
    assert (output / "result.json").read_text() == payload
    assert (output / "stdout.log").read_text() == "experiment finished\n"
    if invalid_run == 1:
        assert not (output.parent / "run-2").exists()
    else:
        assert report["runs"][0]["metrics"] == {"score": 1}


def test_valid_nested_json_and_nonfinite_looking_strings_are_preserved(tmp_path):
    runtime = {
        "environment": {"version": "1e999"},
        "packages": [{"version": "NaN"}, {"version": "Infinity"}],
        "count": 10**400,
        "elapsed": 1e-10,
        "optional": None,
        "enabled": True,
    }
    payload = json.dumps({"metrics": {"score": 0.9}, "runtime": runtime})
    experiment(tmp_path, f"(out / 'result.json').write_text({payload!r})")
    report = verify(tmp_path)
    assert report["status"] == "matched"
    assert [run["runtime"] for run in report["runs"]] == [runtime, runtime]
    saved = json.loads(Path(report["report_path"]).read_text())
    assert saved == report


def test_input_mutation_invalidates_comparison(tmp_path):
    experiment(
        tmp_path,
        """
    Path('train.py').write_text(Path('train.py').read_text() + '\\n# changed')
    (out / 'result.json').write_text(json.dumps({'metrics': {'score': 1}}))
    """,
    )
    result = verify(tmp_path)
    assert result["status"] == "error"
    assert "inputs changed" in result["error"]


def test_timeout_retains_error(tmp_path):
    experiment(tmp_path, "time.sleep(5)", extra="timeout=0.05")
    result = verify(tmp_path)
    assert result["status"] == "error"
    assert "timed out" in result["error"]


def test_missing_inputs_prevent_execution(tmp_path):
    experiment(tmp_path, "Path('started').touch()")
    (tmp_path / "train.py").unlink()
    assert main(["verify", "--root", str(tmp_path)]) == 2
    assert not (tmp_path / "started").exists()


@pytest.mark.parametrize(
    "setting",
    [
        {"metrics": [], "artifacts": []},
        {"timeout": -1},
        {"atol": float("inf")},
        {"command": "python train.py"},
        {"result": "../outside.json"},
    ],
)
def test_invalid_contracts_are_rejected(setting):
    config = {
        "command": ["python", "train.py", "{output}"],
        "inputs": ["train.py"],
        "metrics": ["score"],
        **setting,
    }
    with pytest.raises(ValueError):
        read_verify_config({"verify": config})
