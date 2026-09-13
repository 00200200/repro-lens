from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import tokenize
import tomllib
from pathlib import Path

from .analysis import Finding, analyze

SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".repro-lens",
    ".pytest_cache",
    ".ruff_cache",
}


def inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes project: {path}")
    return resolved


def read_policy(root: Path) -> dict:
    path = root / "pyproject.toml"
    if not path.exists():
        return {}
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    tooling = doc.get("tool", {})
    if not isinstance(tooling, dict):
        raise ValueError("[tool] must be a table")
    policy = tooling.get("repro-lens", {})
    if not isinstance(policy, dict):
        raise ValueError("[tool.repro-lens] must be a table")
    allowed = {"exclude", "required-files", "verify"}
    unknown = set(policy) - allowed
    if unknown:
        raise ValueError(f"Unknown repro-lens settings: {sorted(unknown)}")
    for field in ("exclude", "required-files"):
        value = policy.get(field, [])
        if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
            raise ValueError(f"{field} must be a list of nonempty strings")
    return policy


def paths_to_scan(root: Path, selected: list[str], exclude: list[str]):
    if selected:
        candidates = [Path(path) if Path(path).is_absolute() else root / path for path in selected]
    else:
        try:
            # An ignored directory inside another Git checkout is still a valid project.
            if not (root / ".git").exists():
                raise FileNotFoundError("No Git metadata at the requested project root")
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ],
                capture_output=True,
                check=True,
                timeout=10,
            )
            candidates = [
                root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item
            ]
        except (OSError, subprocess.SubprocessError):
            candidates = []
            for directory, children, files in os.walk(root):
                children[:] = [name for name in children if name not in SKIP]
                candidates.extend(Path(directory) / name for name in files if name.endswith(".py"))
    for path in sorted(set(candidates)):
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            relative = inside(root, path).relative_to(root).as_posix()
        except ValueError:
            continue
        if set(Path(relative).parts) & SKIP or any(fnmatch.fnmatch(relative, x) for x in exclude):
            continue
        yield path, relative


def check(root: Path, selected: list[str] | None = None) -> dict:
    root = root.resolve()
    findings, suppressed = [], []
    try:
        policy = read_policy(root)
    except (ValueError, OSError) as exc:
        policy = {}
        findings.append(
            Finding(
                "pyproject.toml",
                1,
                1,
                "P202",
                "error",
                str(exc),
                "Correct the configuration; it was not applied.",
            )
        )
    for required in policy.get("required-files", []):
        try:
            if not inside(root, root / required).is_file():
                findings.append(
                    Finding(
                        required,
                        1,
                        1,
                        "P201",
                        "error",
                        "File required by this project's policy is missing.",
                        "Create it or deliberately revise the project policy.",
                    )
                )
        except ValueError as exc:
            findings.append(
                Finding(
                    "pyproject.toml",
                    1,
                    1,
                    "P202",
                    "error",
                    str(exc),
                    "Required files must be inside the project.",
                )
            )
    if "verify" in policy:
        from .verify import input_snapshot, read_verify_config

        try:
            config = read_verify_config(policy)
            input_snapshot(root, config["inputs"])
        except (ValueError, OSError) as exc:
            findings.append(
                Finding(
                    "pyproject.toml",
                    1,
                    1,
                    "P203",
                    "error",
                    str(exc),
                    "Correct the verification configuration or restore its inputs.",
                )
            )
    count = 0
    for path, relative in paths_to_scan(root, selected or [], policy.get("exclude", [])):
        count += 1
        try:
            with tokenize.open(path) as handle:
                active, ignored = analyze(handle.read(), relative)
            findings.extend(active)
            suppressed.extend(ignored)
        except (OSError, UnicodeError, SyntaxError) as exc:
            findings.append(
                Finding(
                    relative, 1, 1, "S902", "error", str(exc), "Restore readable Python source."
                )
            )
    findings.sort(key=lambda f: (f.path, f.line, f.code))
    return {
        "schema_version": 1,
        "kind": "static_check",
        "root": str(root),
        "assurance": "Static screening only; experiment repeatability has not been tested.",
        "files_checked": count,
        "findings": [f.to_dict() for f in findings],
        "suppressed": [f.to_dict() for f in suppressed],
        "limitations": [
            "Only documented imported APIs are inspected; wrappers and general data flow "
            "are not resolved.",
            "Explicit seed expressions are accepted but their runtime values are not proven.",
            "Global RNG state, notebooks, training execution and data leakage are not analyzed.",
            "Framework checks cover only the APIs and conditions listed in docs/frameworks.md.",
        ],
    }


def render(report: dict, format_: str) -> str:
    if format_ == "json":
        return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if report["kind"] != "static_check":
        lines = [f"Repro Lens: {report['status']}", report["assurance"]]
        lines += report.get("differences", [])
        if report.get("error"):
            lines.append(report["error"])
        lines.append(f"Evidence: {report.get('report_path', '(stdout)')}")
        return "\n".join(lines) + "\n"
    lines = [
        f"Repro Lens — {report['files_checked']} Python files checked",
        report["assurance"],
        "",
    ]
    for f in report["findings"]:
        line = f"{f['path']}:{f['line']}:{f['column']} {f['code']} [{f['severity']}] {f['message']}"
        lines.extend([f"- {line}" if format_ == "markdown" else line, f"  {f['suggestion']}"])
    if not report["findings"]:
        lines.append("No findings from the enabled checks.")
    if report["suppressed"]:
        lines.append(f"Suppressed findings: {len(report['suppressed'])}")
    return "\n".join(lines) + "\n"
