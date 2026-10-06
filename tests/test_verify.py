import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from repro_lens.cli import main
from repro_lens.project import render_step_summary
from repro_lens.verify import (
    THREADING_ENV_VARS,
    audit_threading_determinism,
    capture_hardware_environment,
    check_stability_bounds,
    compute_metric_statistics,
    digest,
    input_snapshot,
    input_snapshot_path,
    load_input_snapshot,
    read_verify_config,
    verify,
)


def experiment(tmp_path, body, extra="", metrics='["score"]', artifacts="[]", inputs=None):
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
        import json, os, sys, time
        from pathlib import Path
        out = Path(sys.argv[1])
    """)
        + textwrap.dedent(body)
    )
    inputs = inputs or '["train.py", "pyproject.toml"]'
    (tmp_path / "pyproject.toml").write_text(f"""
[tool.repro-lens.verify]
command = ["{{python}}", "train.py", "{{output}}"]
inputs = {inputs}
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
        ("(out / 'result.json').write_text('{\"metrics\": {\"score\": 1e-400}}')", "Underflowing"),
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
        ('{"metrics":{"score":1e-400}}', "Underflowing JSON number"),
        ('{"metrics":{"score":-1e-400}}', "Underflowing JSON number"),
        ('{"metrics":{"score":1},"runtime":{"elapsed":1e-400}}', "Underflowing JSON number"),
        ('{"metrics":{"score":1,"unused":1e-400}}', "Underflowing JSON number"),
        ('{"metrics":{"score":1},"runtime":{"samples":[1e-400]}}', "Underflowing JSON number"),
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
        "underflowing-metric",
        "negative-underflow",
        "underflowing-runtime",
        "undeclared-underflow",
        "underflow-in-array",
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


def test_zero_exponents_are_not_treated_as_underflow(tmp_path):
    experiment(
        tmp_path,
        "(out / 'result.json').write_text('{\"metrics\": {\"score\": 0e-400}}')",
    )
    result = verify(tmp_path)
    assert result["status"] == "matched"
    assert [run["metrics"]["score"] for run in result["runs"]] == [0.0, 0.0]


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


def test_sha256_snapshot_detects_silent_input_edits_across_verifies(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    experiment(
        tmp_path,
        "(out / 'result.json').write_text(json.dumps({'metrics': {'score': 1}}))",
        inputs='["train.py", "pyproject.toml", "data.csv"]',
        extra="hash-inputs = true\n",
    )
    first = verify(tmp_path)
    assert first["status"] == "matched"
    snapshot = input_snapshot_path(tmp_path)
    assert Path(first["input_snapshot_path"]) == snapshot
    stored = load_input_snapshot(tmp_path)
    assert stored == first["inputs_sha256"]
    assert stored["data.csv"] == digest(tmp_path / "data.csv")

    second = verify(tmp_path)
    assert second["status"] == "matched"
    assert second["inputs_sha256"] == first["inputs_sha256"]

    (tmp_path / "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    third = verify(tmp_path)
    assert third["status"] == "error"
    assert "previous SHA-256 snapshot" in third["error"]
    assert "data.csv" in third["error"]
    assert load_input_snapshot(tmp_path) == first["inputs_sha256"]


def test_hash_inputs_false_skips_persisted_snapshot(tmp_path):
    (tmp_path / "data.csv").write_text("row\n", encoding="utf-8")
    experiment(
        tmp_path,
        "(out / 'result.json').write_text(json.dumps({'metrics': {'score': 1}}))",
        inputs='["train.py", "pyproject.toml", "data.csv"]',
        extra="hash-inputs = false\n",
    )
    first = verify(tmp_path)
    assert first["status"] == "matched"
    assert "input_snapshot_path" not in first
    assert not input_snapshot_path(tmp_path).exists()
    assert first["inputs_sha256"]["data.csv"] == digest(tmp_path / "data.csv")

    (tmp_path / "data.csv").write_text("row\nchanged\n", encoding="utf-8")
    second = verify(tmp_path)
    assert second["status"] == "matched"
    assert second["inputs_sha256"]["data.csv"] != first["inputs_sha256"]["data.csv"]


def test_input_snapshot_hashes_small_temp_files(tmp_path):
    payload = b"tiny-fixture\n"
    path = tmp_path / "fixture.bin"
    path.write_bytes(payload)
    hashes = input_snapshot(tmp_path, ["fixture.bin"])
    assert hashes == {"fixture.bin": hashlib.sha256(payload).hexdigest()}


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
        {"hash-inputs": "yes"},
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


def test_capture_hardware_environment():
    env = capture_hardware_environment()
    assert "runner_python" in env
    assert "platform" in env
    assert "machine" in env
    assert "processor" in env
    assert "cpu_count" in env
    assert "cuda_available" in env
    assert isinstance(env["cuda_available"], bool)
    assert isinstance(env["platform"], str) and len(env["platform"]) > 0


def test_verify_report_retains_hardware_environment(tmp_path):
    experiment(
        tmp_path,
        """
    (out / 'result.json').write_text(json.dumps({'metrics': {'score': 0.9}}))
    """,
    )
    result = verify(tmp_path)
    assert result["status"] == "matched"
    assert "environment" in result
    env = result["environment"]
    assert "runner_python" in env
    assert "platform" in env
    assert "machine" in env
    assert "cuda_available" in env
    assert "processor" in env
    assert "threading" in env


def test_audit_threading_determinism_all_pinned():
    env = {var: "1" for var in THREADING_ENV_VARS}
    warnings = audit_threading_determinism(env)
    assert warnings == []


def test_audit_threading_determinism_unpinned():
    warnings = audit_threading_determinism({})
    assert len(warnings) == 1
    assert "OMP_NUM_THREADS" in warnings[0]
    assert "MKL_NUM_THREADS" in warnings[0]
    assert "OPENBLAS_NUM_THREADS" in warnings[0]
    assert "VECLIB_MAXIMUM_THREADS" in warnings[0]
    assert "NUMEXPR_NUM_THREADS" in warnings[0]
    assert "Recommend explicit pinning" in warnings[0]


def test_audit_threading_determinism_dynamic_or_invalid():
    env = {
        "OMP_NUM_THREADS": "dynamic",
        "MKL_NUM_THREADS": "0",
        "OPENBLAS_NUM_THREADS": "-2",
        "VECLIB_MAXIMUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "2",
    }
    warnings = audit_threading_determinism(env)
    assert any("OMP_NUM_THREADS='dynamic'" in w for w in warnings)
    assert any("MKL_NUM_THREADS='0'" in w for w in warnings)
    assert any("OPENBLAS_NUM_THREADS='-2'" in w for w in warnings)


def test_verify_report_retains_warnings(tmp_path):
    experiment(
        tmp_path,
        """
    (out / 'result.json').write_text(json.dumps({'metrics': {'score': 0.9}}))
    """,
    )
    result = verify(tmp_path)
    assert "warnings" in result
    assert isinstance(result["warnings"], list)


def test_compute_metric_statistics():
    empty_stats = compute_metric_statistics([])
    assert empty_stats["count"] == 0
    assert empty_stats["mean"] == 0.0

    single_stats = compute_metric_statistics([42.0])
    assert single_stats["count"] == 1
    assert single_stats["mean"] == 42.0
    assert single_stats["variance"] == 0.0
    assert single_stats["std"] == 0.0

    stats = compute_metric_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["count"] == 5
    assert stats["mean"] == 3.0
    assert stats["variance"] == 2.5
    assert abs(stats["std"] - 1.581139) < 1e-5
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0


def test_check_stability_bounds():
    stats = {
        "accuracy": {"mean": 0.88, "std": 0.015, "variance": 0.000225},
        "loss": {"mean": 0.35, "std": 0.05, "variance": 0.0025},
    }
    # Within bounds
    failures = check_stability_bounds(
        stats,
        {"accuracy": {"max_std": 0.02, "min_mean": 0.85}, "loss": {"max_mean": 0.50}},
    )
    assert failures == []

    # Exceeding std
    failures = check_stability_bounds(stats, {"accuracy": {"max_std": 0.01}})
    assert len(failures) == 1
    assert "accuracy" in failures[0]
    assert "exceeds max_std" in failures[0]

    # Below min_mean
    failures = check_stability_bounds(stats, {"accuracy": {"min_mean": 0.90}})
    assert len(failures) == 1
    assert "below min_mean" in failures[0]


def test_multi_seed_verification_stable(tmp_path):
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
        import json, os, sys
        from pathlib import Path
        out = Path(sys.argv[1])
        seed = int(os.environ.get("SEED", sys.argv[2] if len(sys.argv) > 2 else "0"))
        # Slight variation across seeds
        score = 0.90 + (seed % 5) * 0.001
        (out / 'result.json').write_text(json.dumps({'metrics': {'score': score}}))
    """)
    )
    (tmp_path / "pyproject.toml").write_text("""
[tool.repro-lens.verify]
command = ["{python}", "train.py", "{output}", "{seed}"]
inputs = ["train.py", "pyproject.toml"]
metrics = ["score"]
artifacts = []

[tool.repro-lens.verify.stability]
score = { max_std = 0.05, min_mean = 0.85 }
""")
    result = verify(tmp_path, seeds=[42, 43, 44])
    assert result["status"] == "stable"
    assert result["kind"] == "multi_seed_verification"
    assert result["seeds"] == [42, 43, 44]
    assert len(result["runs"]) == 3
    assert result["stability_failures"] == []
    assert "score" in result["statistics"]
    assert result["statistics"]["score"]["count"] == 3


def test_multi_seed_verification_unstable(tmp_path):
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
        import json, os, sys
        from pathlib import Path
        out = Path(sys.argv[1])
        seed = int(os.environ.get("SEED", sys.argv[2] if len(sys.argv) > 2 else "0"))
        # High variance across seeds
        score = 0.50 if seed == 42 else 0.95
        (out / 'result.json').write_text(json.dumps({'metrics': {'score': score}}))
    """)
    )
    (tmp_path / "pyproject.toml").write_text("""
[tool.repro-lens.verify]
command = ["{python}", "train.py", "{output}", "{seed}"]
inputs = ["train.py", "pyproject.toml"]
metrics = ["score"]
artifacts = []

[tool.repro-lens.verify.stability]
score = { max_std = 0.05 }
""")
    result = verify(tmp_path, seeds=[42, 43])
    assert result["status"] == "unstable"
    assert len(result["stability_failures"]) == 1
    assert "exceeds max_std" in result["stability_failures"][0]


def test_cli_multi_seed_verify(tmp_path, capsys):
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
        import json, os, sys
        from pathlib import Path
        out = Path(sys.argv[1])
        seed = int(os.environ.get("SEED", "0"))
        score = 0.88 + seed * 0.001
        (out / 'result.json').write_text(json.dumps({'metrics': {'score': score}}))
    """)
    )
    (tmp_path / "pyproject.toml").write_text("""
[tool.repro-lens.verify]
command = ["{python}", "train.py", "{output}"]
inputs = ["train.py", "pyproject.toml"]
metrics = ["score"]
artifacts = []

[tool.repro-lens.verify.stability]
score = { max_std = 0.10 }
""")
    # Stable run
    code = main(["verify", "--root", str(tmp_path), "--seeds", "1,2,3"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Repro Lens: stable" in captured.out
    assert "Metric 'score'" in captured.out

    # Invalid seeds format
    err_code = main(["verify", "--root", str(tmp_path), "--seeds", "abc,def"])
    assert err_code == 2
    captured = capsys.readouterr()
    assert "Invalid --seeds format" in captured.err

    # Fewer than 2 seeds
    err_code = main(["verify", "--root", str(tmp_path), "--seeds", "42"])
    assert err_code == 2
    captured = capsys.readouterr()
    assert "requires at least 2 seeds" in captured.err


def test_multi_seed_step_summary(tmp_path):
    report = {
        "kind": "multi_seed_verification",
        "status": "stable",
        "assurance": "Test assurance",
        "seeds": [1, 2, 3],
        "statistics": {
            "accuracy": {
                "count": 3,
                "mean": 0.95,
                "std": 0.01,
                "variance": 0.0001,
                "min": 0.94,
                "max": 0.96,
            }
        },
        "stability_failures": [],
    }
    summary = render_step_summary(report)
    assert "## Repro Lens — `stable`" in summary
    assert "Metric Stability Across Seeds" in summary
    assert "| `accuracy` | 0.95 | 0.01 | 0.0001 | 0.94 | 0.96 |" in summary
    assert "All metrics satisfied stability bounds." in summary


def test_read_verify_config_sandbox_validation():
    base = {
        "command": ["{python}", "train.py", "{output}"],
        "inputs": ["train.py"],
        "metrics": ["score"],
    }
    # Valid docker
    cfg = read_verify_config(
        {"verify": {**base, "sandbox": "docker", "sandbox-image": "custom:1.0"}}
    )
    assert cfg["sandbox"] == "docker"
    assert cfg["sandbox-image"] == "custom:1.0"

    # Valid podman
    cfg2 = read_verify_config({"verify": {**base, "sandbox": "podman"}})
    assert cfg2["sandbox"] == "podman"
    assert cfg2["sandbox-image"] is None

    # Invalid engine
    with pytest.raises(ValueError, match="verify.sandbox must be 'docker' or 'podman'"):
        read_verify_config({"verify": {**base, "sandbox": "containerd"}})

    # Invalid empty image
    with pytest.raises(ValueError, match="verify.sandbox-image must be a nonempty string"):
        read_verify_config({"verify": {**base, "sandbox-image": ""}})


def test_sandbox_missing_binary_raises(tmp_path, monkeypatch):
    from repro_lens.verify import execute

    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ValueError, match="Sandbox engine 'docker' not found in PATH"):
        execute(
            ["python3", "train.py", "{output}"],
            tmp_path,
            tmp_path / "run-1",
            timeout=10,
            sandbox="docker",
        )


def test_sandbox_execution_and_reporting(tmp_path, monkeypatch):
    import subprocess

    from repro_lens.project import render
    from repro_lens.verify import verify

    experiment(
        tmp_path,
        """
    (out / 'result.json').write_text(json.dumps({'metrics': {'score': 0.95}}))
    """,
    )

    # Mock subprocess.Popen and shutil.which
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    captured_cmds = []
    orig_popen = subprocess.Popen

    def mock_popen(argv, *args, **kwargs):
        if argv and argv[0] == "docker":
            captured_cmds.append(argv)
            for arg in argv:
                if "/workspace/.repro-lens/verify" in arg:
                    rel_part = arg.replace("/workspace/", "")
                    out_path = tmp_path / rel_part
                    out_path.mkdir(parents=True, exist_ok=True)
                    (out_path / "result.json").write_text(json.dumps({"metrics": {"score": 0.95}}))
            return orig_popen(["python3", "-c", "import sys; sys.exit(0)"], *args, **kwargs)
        return orig_popen(argv, *args, **kwargs)

    monkeypatch.setattr("subprocess.Popen", mock_popen)

    res = verify(tmp_path, sandbox="docker", sandbox_image="python:3.11-slim")
    assert res["status"] == "matched"
    assert res["sandbox"] == "docker"
    assert res["sandbox_image"] == "python:3.11-slim"
    assert len(captured_cmds) == 2
    for cmd in captured_cmds:
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        assert f"{tmp_path.resolve()}:/workspace" in cmd[cmd.index("-v") + 1]
        assert "python:3.11-slim" in cmd

    rendered = render(res, "text")
    assert "Sandbox: docker (python:3.11-slim)" in rendered
    summary = render_step_summary(res)
    assert "| Sandbox | `docker` (`python:3.11-slim`) |" in summary
