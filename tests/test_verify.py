import json
import textwrap

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
    from pathlib import Path

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
        ("(out / 'result.json').write_text('{\"metrics\": {\"score\": true}}')", "nonnumeric"),
        ("(out / 'result.json').write_text('bad json')", "Expecting"),
    ],
)
def test_failed_or_invalid_results_never_pass(tmp_path, body, reason):
    experiment(tmp_path, body)
    result = verify(tmp_path)
    assert result["status"] == "error"
    assert reason in result["error"]


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
