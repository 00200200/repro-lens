"""Compare retained verification evidence without executing or importing a project."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .json_data import loads, same_json
from .verify import output_differences, read_verify_config

REPORT_LIMIT = 8_000_000
CONFIG_FIELDS = {"command", "inputs", "metrics", "artifacts", "timeout", "atol", "rtol", "result"}


def hash_map(value: object, field: str) -> dict:
    if not isinstance(value, dict) or any(
        not name or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
        for name, digest in value.items()
    ):
        raise ValueError(f"{field} must map nonempty names to SHA-256 hashes")
    return value


def read_report(path: Path) -> tuple[dict, str]:
    with path.open("rb") as handle:
        raw = handle.read(REPORT_LIMIT + 1)
    if len(raw) > REPORT_LIMIT:
        raise ValueError(f"Report exceeds {REPORT_LIMIT} bytes: {path}")
    report = loads(raw.decode("utf-8"), path)
    if not isinstance(report, dict):
        raise ValueError(f"Expected a verification report object: {path}")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError(f"Unsupported report schema_version: {path}")
    if report.get("kind") != "repeatability_test":
        raise ValueError(f"Expected a repeatability_test report: {path}")
    if report.get("status") != "matched":
        raise ValueError(f"Both reports must have matched status: {path}")
    if report.get("differences") != [] or "error" in report:
        raise ValueError(f"Matched report contains errors or missing/nonempty differences: {path}")
    original = report.get("configuration")
    if not isinstance(original, dict) or set(original) != CONFIG_FIELDS:
        raise ValueError(f"Report must retain the complete verification configuration: {path}")
    try:
        report["configuration"] = read_verify_config({"verify": original})
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid report configuration in {path}: {exc}") from exc
    config = report["configuration"]
    inputs = hash_map(report.get("inputs_sha256"), "inputs_sha256")
    if not inputs:
        raise ValueError(f"Report has no declared input hashes: {path}")
    environment = report.get("environment")
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(key), str) or not environment[key]
        for key in ("runner_python", "platform", "machine")
    ):
        raise ValueError(f"Report must retain its runner environment: {path}")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError(f"Report must contain exactly two successful runs: {path}")
    for run in runs:
        if (
            not isinstance(run, dict)
            or type(run.get("returncode")) is not int
            or run["returncode"] != 0
            or "runtime" not in run
        ):
            raise ValueError(f"Report contains an incomplete or unsuccessful run: {path}")
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(config["metrics"]):
            raise ValueError(f"Run metrics must match declared metric names: {path}")
        if any(
            type(value) not in (int, float) or (type(value) is float and not math.isfinite(value))
            for value in metrics.values()
        ):
            raise ValueError(f"Run metrics must be finite numbers: {path}")
        artifacts = hash_map(run.get("artifacts_sha256"), "artifacts_sha256")
        if set(artifacts) != set(config["artifacts"]):
            raise ValueError(f"Run hashes must match declared artifact names: {path}")
    if output_differences(*runs, config):
        raise ValueError(f"Recorded run values contradict matched status: {path}")
    return report, hashlib.sha256(raw).hexdigest()


def changed_fields(before: dict, after: dict, *, metadata: bool = False) -> dict:
    return {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in sorted(before.keys() | after.keys())
        if name not in before
        or name not in after
        or (not same_json(before[name], after[name]) if metadata else before[name] != after[name])
    }


def compare_reports(before_path: Path, after_path: Path) -> dict:
    before, before_hash = read_report(before_path)
    after, after_hash = read_report(after_path)
    before_inputs, after_inputs = before["inputs_sha256"], after["inputs_sha256"]
    policy_changes = changed_fields(before["configuration"], after["configuration"])
    environment_changes = changed_fields(before["environment"], after["environment"], metadata=True)
    first_runtime, second_runtime = before["runs"][0]["runtime"], after["runs"][0]["runtime"]
    if not same_json(first_runtime, second_runtime):
        environment_changes["experiment_runtime"] = {
            "before": first_runtime,
            "after": second_runtime,
        }
    report = {
        "schema_version": 1,
        "kind": "report_comparison",
        "status": "not_comparable" if policy_changes or environment_changes else "matched",
        "assurance": (
            "Recorded outputs only; changed inputs need review. "
            "Reports are not authenticated and no experiment is executed."
        ),
        "before": {
            "path": str(before_path.resolve()),
            "sha256": before_hash,
            "git": before.get("git"),
        },
        "after": {
            "path": str(after_path.resolve()),
            "sha256": after_hash,
            "git": after.get("git"),
        },
        "policy_changes": policy_changes,
        "environment_changes": environment_changes,
        "input_changes": {
            "added": sorted(after_inputs.keys() - before_inputs.keys()),
            "removed": sorted(before_inputs.keys() - after_inputs.keys()),
            "modified": sorted(
                name
                for name in before_inputs.keys() & after_inputs.keys()
                if before_inputs[name] != after_inputs[name]
            ),
        },
        "differences": [],
    }
    if report["status"] == "not_comparable":
        return report
    for left, first in enumerate(before["runs"], 1):
        for right, second in enumerate(after["runs"], 1):
            report["differences"].extend(
                f"Before run {left}, after run {right}: {difference}"
                for difference in output_differences(first, second, before["configuration"])
            )
    if report["differences"]:
        report["status"] = "mismatch"
    return report


def render_comparison(report: dict, format_: str) -> str:
    if format_ == "json":
        return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    lines = [f"Repro Lens: {report['status']}", report["assurance"]]
    lines.extend(f"{side.title()}: {report[side]['path']}" for side in ("before", "after"))
    for category in ("policy_changes", "environment_changes"):
        for name in report[category]:
            lines.append(f"{category}: {name}")
    for change, names in report["input_changes"].items():
        lines.extend(f"Input {change}: {name}" for name in names)
    lines.extend(report["differences"])
    return "\n".join(lines) + "\n"
