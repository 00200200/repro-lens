"""Run an explicitly requested local experiment twice and retain bounded evidence.

This runner is not a sandbox. It executes a reviewed argv, never a shell string.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from fractions import Fraction
from pathlib import Path

from .json_data import loads
from .project import inside, read_policy


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_verify_config(policy: dict) -> dict:
    original = policy.get("verify")
    if not isinstance(original, dict):
        raise ValueError("Configure [tool.repro-lens.verify] before running verify")
    config = original.copy()
    allowed = {"command", "inputs", "metrics", "artifacts", "timeout", "atol", "rtol", "result"}
    if set(config) - allowed:
        raise ValueError(f"Unknown verify settings: {sorted(set(config) - allowed)}")
    for field in ("command", "inputs", "metrics", "artifacts"):
        value = config.setdefault(field, [])
        if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
            raise ValueError(f"verify.{field} must be a list of nonempty strings")
    if not config["command"] or not any("{output}" in arg for arg in config["command"]):
        raise ValueError("verify.command must be an argv list containing {output}")
    if not config["inputs"]:
        raise ValueError("verify.inputs must explicitly identify code, configuration and data")
    if not config["metrics"] and not config["artifacts"]:
        raise ValueError("verify requires at least one metric or artifact to compare")
    for field, default in (("timeout", 60), ("atol", 0.0), ("rtol", 0.0)):
        value = config.setdefault(field, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"verify.{field} must be finite and nonnegative")
    if config["timeout"] == 0:
        raise ValueError("verify.timeout must be positive")
    config.setdefault("result", "result.json")
    for value in [config["result"], *config["artifacts"]]:
        if (
            not isinstance(value, str)
            or not value
            or Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise ValueError("Result and artifact names must be relative paths without '..'")
    return config


def input_snapshot(root: Path, patterns: list[str]) -> dict[str, str]:
    hashes = {}
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"Input pattern must stay inside the project: {pattern}")
        paths = [path for path in root.glob(pattern) if path.is_file()]
        if not paths:
            raise ValueError(f"Input pattern matched no files: {pattern}")
        for path in paths:
            safe = inside(root, path)
            hashes[path.relative_to(root).as_posix()] = digest(safe)
    return dict(sorted(hashes.items()))


def git_state(root: Path) -> dict:
    def run(args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        status = run(["status", "--porcelain", "--untracked-files=normal"])
        return {
            "commit": run(["rev-parse", "HEAD"]),
            "dirty": bool(status) if status is not None else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def execute(command: list[str], root: Path, run_dir: Path, timeout: float) -> dict:
    run_dir.mkdir()
    argv = [
        arg.replace("{output}", str(run_dir)).replace("{python}", sys.executable) for arg in command
    ]
    start = time.monotonic()
    # Logs go to files so experiment output cannot grow the runner's memory unboundedly.
    with (run_dir / "stdout.log").open("wb") as out, (run_dir / "stderr.log").open("wb") as err:
        process = subprocess.Popen(
            argv, cwd=root, stdout=out, stderr=err, start_new_session=os.name == "posix"
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise ValueError(f"Experiment timed out after {timeout}s; see {run_dir}") from None
    evidence = {
        "argv": argv,
        "returncode": returncode,
        "elapsed_seconds": round(time.monotonic() - start, 4),
        "output": str(run_dir),
    }
    if returncode:
        raise ValueError(f"Experiment exited with {returncode}; see {run_dir / 'stderr.log'}")
    return evidence


def load_result(run_dir: Path, config: dict) -> dict:
    path = inside(run_dir, run_dir / config["result"])
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError(f"Missing or oversized result JSON: {path}")

    result = loads(path.read_text(encoding="utf-8"), path)
    if not isinstance(result, dict) or not isinstance(result.get("metrics", {}), dict):
        raise ValueError(f"Result must be an object with a metrics object: {path}")
    for metric in config["metrics"]:
        value = result.get("metrics", {}).get(metric)
        if type(value) not in (float, int) or (type(value) is float and not math.isfinite(value)):
            raise ValueError(f"Missing, nonnumeric or nonfinite metric {metric!r}: {path}")
    for artifact in config["artifacts"]:
        if not inside(run_dir, run_dir / artifact).is_file():
            raise ValueError(f"Missing artifact: {artifact}")
    return result


def metrics_close(first: int | float, second: int | float, *, atol: float, rtol: float) -> bool:
    # Preserve integer precision and avoid overflow in the tolerance calculation.
    a, b = Fraction(first), Fraction(second)
    threshold = max(Fraction(atol), Fraction(rtol) * max(abs(a), abs(b)))
    return abs(a - b) <= threshold


def output_differences(first: dict, second: dict, config: dict) -> list[str]:
    differences = []
    for name in config["metrics"]:
        a, b = first["metrics"][name], second["metrics"][name]
        if not metrics_close(a, b, rtol=config["rtol"], atol=config["atol"]):
            differences.append(f"Metric {name}: {a} != {b}")
    for name in config["artifacts"]:
        if first["artifacts_sha256"][name] != second["artifacts_sha256"][name]:
            differences.append(f"Artifact {name}: SHA-256 differs")
    if first["runtime"] != second["runtime"]:
        differences.append("Reported experiment runtimes differ")
    return differences


def verify(root: Path) -> dict:
    root = root.resolve()
    config = read_verify_config(read_policy(root))
    before = input_snapshot(root, config["inputs"])
    state = git_state(root)
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    evidence_dir = inside(root, root / ".repro-lens" / "verify" / run_id)
    evidence_dir.mkdir(parents=True)
    report_path = evidence_dir / "report.json"
    report = {
        "schema_version": 1,
        "kind": "repeatability_test",
        "status": "error",
        "assurance": (
            "Two runs test only the declared outputs in this local environment; "
            "not scientific validity or cross-platform reproducibility."
        ),
        "root": str(root),
        "git": state,
        "inputs_sha256": before,
        "environment": {
            "runner_python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "configuration": config,
        "runs": [],
        "differences": [],
        "report_path": str(report_path),
    }
    try:
        outputs = []
        for index in (1, 2):
            run_dir = evidence_dir / f"run-{index}"
            execution = execute(config["command"], root, run_dir, config["timeout"])
            report["runs"].append(execution)
            result = load_result(run_dir, config)
            # Keep only configured metrics and bounded runtime metadata in the report.
            execution["metrics"] = {name: result["metrics"][name] for name in config["metrics"]}
            execution["runtime"] = result.get("runtime", {})
            execution["artifacts_sha256"] = {
                name: digest(inside(run_dir, run_dir / name)) for name in config["artifacts"]
            }
            outputs.append(execution)
            if input_snapshot(root, config["inputs"]) != before:
                raise ValueError(
                    "Declared inputs changed during verification; the comparison is invalid"
                )
        report["differences"] = output_differences(*outputs, config)
        report["status"] = "mismatch" if report["differences"] else "matched"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
