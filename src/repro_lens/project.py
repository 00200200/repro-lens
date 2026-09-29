from __future__ import annotations

import fnmatch
import importlib.metadata
import json
import os
import re
import subprocess
import tokenize
import tomllib
from dataclasses import replace
from pathlib import Path

from .analysis import Finding, analyze
from .notebooks import notebook_source

LOCKFILE_NAMES = ("uv.lock", "poetry.lock")

SUFFIXES = {".py", ".ipynb"}
SKIP = {
    ".git",
    ".ipynb_checkpoints",
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


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def find_lockfile(root: Path) -> Path | None:
    for name in LOCKFILE_NAMES:
        path = root / name
        if path.is_file():
            return path
    return None


def read_lockfile_versions(path: Path) -> dict[str, set[str]]:
    """Parse uv.lock or poetry.lock into normalized name -> allowed versions."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Lockfile is not valid TOML: {path.name}: {exc}") from exc
    packages = document.get("package")
    if not isinstance(packages, list):
        raise ValueError(f"Lockfile has no package table: {path.name}")
    locked: dict[str, set[str]] = {}
    for entry in packages:
        if not isinstance(entry, dict):
            raise ValueError(f"Lockfile package entry must be a table: {path.name}")
        name, version = entry.get("name"), entry.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError(f"Lockfile package entries need name and version: {path.name}")
        locked.setdefault(normalize_package_name(name), set()).add(version)
    return locked


def installed_package_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if isinstance(name, str) and name and isinstance(version, str) and version:
            installed[normalize_package_name(name)] = version
    return installed


def lockfile_sync_mismatches(locked: dict[str, set[str]], installed: dict[str, str]) -> list[str]:
    """Compare a lockfile map to an installed package map; ignore packages only on one side."""
    mismatches = []
    for name in sorted(set(locked) & set(installed)):
        version = installed[name]
        allowed = locked[name]
        if version not in allowed:
            expected = ", ".join(sorted(allowed))
            mismatches.append(f"{name} is {version} but lockfile has {expected}")
    return mismatches


def lockfile_environment_mismatches(
    root: Path, installed: dict[str, str] | None = None
) -> tuple[str | None, list[str]]:
    """Return (lockfile name or None, mismatch messages) for the project's lockfile."""
    path = find_lockfile(root)
    if path is None:
        return None, []
    locked = read_lockfile_versions(path)
    packages = installed if installed is not None else installed_package_versions()
    return path.name, lockfile_sync_mismatches(locked, packages)


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
                candidates.extend(
                    Path(directory) / name for name in files if Path(name).suffix in SUFFIXES
                )
    for path in sorted(set(candidates)):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        try:
            relative = inside(root, path).relative_to(root).as_posix()
        except ValueError:
            continue
        if set(Path(relative).parts) & SKIP or any(fnmatch.fnmatch(relative, x) for x in exclude):
            continue
        yield path, relative


def analyze_notebook(path: Path, relative: str) -> tuple[list[Finding], list[Finding]]:
    source, locations, invalid = notebook_source(path.read_text(encoding="utf-8"))
    active, ignored = analyze(source, relative)

    def in_cell(finding: Finding) -> Finding:
        if not 1 <= finding.line <= len(locations):
            return finding  # A whole-notebook error without a cell location.
        cell, line = locations[finding.line - 1]
        return replace(finding, cell=cell, line=line)

    errors = [
        Finding(
            relative,
            line,
            1,
            "S902",
            "error",
            f"Cell is not valid Python and was not checked: {message}",
            "Fix the cell or exclude the notebook; the other cells were still checked.",
            cell=cell,
        )
        for cell, line, message in invalid
    ]
    return [in_cell(f) for f in active] + errors, [in_cell(f) for f in ignored]


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
        try:
            lockfile_name, mismatches = lockfile_environment_mismatches(root)
        except (ValueError, OSError) as exc:
            findings.append(
                Finding(
                    "pyproject.toml",
                    1,
                    1,
                    "P204",
                    "error",
                    str(exc),
                    "Repair the lockfile or regenerate it with uv lock / poetry lock.",
                )
            )
        else:
            for message in mismatches:
                findings.append(
                    Finding(
                        lockfile_name or "uv.lock",
                        1,
                        1,
                        "P204",
                        "error",
                        message,
                        "Sync the environment to the lockfile (uv sync --locked) before verify.",
                    )
                )
    count = 0
    for path, relative in paths_to_scan(root, selected or [], policy.get("exclude", [])):
        count += 1
        try:
            if path.suffix == ".ipynb":
                active, ignored = analyze_notebook(path, relative)
            else:
                with tokenize.open(path) as handle:
                    active, ignored = analyze(handle.read(), relative)
            findings.extend(active)
            suppressed.extend(ignored)
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            findings.append(
                Finding(
                    relative, 1, 1, "S902", "error", str(exc), "Restore readable Python source."
                )
            )
    findings.sort(key=lambda f: (f.path, f.cell or 0, f.line, f.code))
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
            "Notebook code cells are read in file order; IPython magics and shell lines are "
            "skipped, and execution order and outputs are not analyzed.",
            "Global RNG seeding is recognized only within the same file; training execution "
            "and data leakage are not analyzed.",
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
        f"Repro Lens — {report['files_checked']} Python files and notebooks checked",
        report["assurance"],
        "",
    ]
    for f in report["findings"]:
        where = f"{f['path']}:cell {f['cell']}" if "cell" in f else f["path"]
        line = f"{where}:{f['line']}:{f['column']} {f['code']} [{f['severity']}] {f['message']}"
        lines.extend([f"- {line}" if format_ == "markdown" else line, f"  {f['suggestion']}"])
    if not report["findings"]:
        lines.append("No findings from the enabled checks.")
    if report["suppressed"]:
        lines.append(f"Suppressed findings: {len(report['suppressed'])}")
    return "\n".join(lines) + "\n"
