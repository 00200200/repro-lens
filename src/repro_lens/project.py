from __future__ import annotations

import fnmatch
import importlib.metadata
import json
import os
import re
import subprocess
import tokenize
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .analysis import CustomRule, Finding, analyze
from .frameworks import RULES as FRAMEWORK_RULES
from .notebooks import notebook_source

LOCKFILE_NAMES = ("uv.lock", "poetry.lock")
# Rule codes a suppression comment may list.
SUPPRESSIBLE_CODES = {"R101", "R102", "R103", "R190", *FRAMEWORK_RULES}
IGNORE_JUSTIFICATION = "TODO: Review reproducibility"
_EXISTING_IGNORE = re.compile(r"#\s*repro-lens:\s*ignore")

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


def _fallback_parse_rules_yaml(text: str) -> list[dict]:
    """Basic fallback parser for .repro-lens/rules.yaml when PyYAML is unavailable."""
    rules = []
    current_rule = None
    in_match = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith(("- id:", "- id :")):
            if current_rule:
                rules.append(current_rule)
            current_rule = {
                "id": stripped.split(":", 1)[1].strip().strip("\"'"),
                "match": {},
            }
            in_match = False
            continue

        if current_rule is None:
            continue

        if stripped.startswith("id:"):
            current_rule["id"] = stripped.split(":", 1)[1].strip().strip("\"'")
        elif stripped.startswith("severity:"):
            current_rule["severity"] = stripped.split(":", 1)[1].strip().strip("\"'")
        elif stripped.startswith("message:"):
            current_rule["message"] = stripped.split(":", 1)[1].strip().strip("\"'")
        elif stripped.startswith("suggestion:"):
            current_rule["suggestion"] = stripped.split(":", 1)[1].strip().strip("\"'")
        elif stripped.startswith("match:"):
            in_match = True
        elif in_match and stripped.startswith("call:"):
            current_rule["match"]["call"] = stripped.split(":", 1)[1].strip().strip("\"'")
        elif in_match and stripped.startswith("missing_kwargs:"):
            raw_val = stripped.split(":", 1)[1].strip()
            if raw_val.startswith("[") and raw_val.endswith("]"):
                items = [k.strip().strip("\"'") for k in raw_val[1:-1].split(",") if k.strip()]
                current_rule["match"]["missing_kwargs"] = items

    if current_rule:
        rules.append(current_rule)
    return rules


def parse_custom_rules_yaml(text: str) -> list[CustomRule]:
    """Parse custom rule definitions from YAML text."""
    rules_data = None
    try:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            rules_data = data.get("rules")
    except ImportError:
        pass

    if rules_data is None:
        rules_data = _fallback_parse_rules_yaml(text)

    if not isinstance(rules_data, list):
        return []

    result: list[CustomRule] = []
    for item in rules_data:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("id") or "").strip()
        if not rule_id:
            continue
        msg = str(item.get("message") or f"Custom rule {rule_id} violated")
        sev = str(item.get("severity") or "warning").lower()
        sug = str(item.get("suggestion") or "")
        match_spec = item.get("match") or {}
        call = str(match_spec.get("call") or "").strip()
        missing_kwargs = match_spec.get("missing_kwargs") or ()
        if isinstance(missing_kwargs, str):
            missing_kwargs = (missing_kwargs,)
        elif isinstance(missing_kwargs, list):
            missing_kwargs = tuple(str(k) for k in missing_kwargs)

        result.append(
            CustomRule(
                id=rule_id,
                message=msg,
                severity=sev,
                suggestion=sug,
                call=call,
                missing_kwargs=missing_kwargs,
            )
        )
    return result


def load_custom_rules(root: Path) -> list[CustomRule]:
    """Load custom rule definitions from .repro-lens/rules.yaml or .repro-lens/rules.yml."""
    for filename in ("rules.yaml", "rules.yml"):
        rule_path = root / ".repro-lens" / filename
        if rule_path.is_file():
            try:
                return parse_custom_rules_yaml(rule_path.read_text(encoding="utf-8"))
            except Exception:
                return []
    return []


def analyze_notebook(
    path: Path,
    relative: str,
    custom_rules: tuple[CustomRule, ...] | list[CustomRule] = (),
) -> tuple[list[Finding], list[Finding]]:
    source, locations, invalid = notebook_source(path.read_text(encoding="utf-8"))
    active, ignored = analyze(source, relative, custom_rules=custom_rules)

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
    custom_rules = load_custom_rules(root)
    count = 0
    for path, relative in paths_to_scan(root, selected or [], policy.get("exclude", [])):
        count += 1
        try:
            if path.suffix == ".ipynb":
                active, ignored = analyze_notebook(path, relative, custom_rules=custom_rules)
            else:
                with tokenize.open(path) as handle:
                    active, ignored = analyze(handle.read(), relative, custom_rules=custom_rules)
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


def ignore_comment(codes: set[str]) -> str:
    return f"# repro-lens: ignore[{', '.join(sorted(codes))}] -- {IGNORE_JUSTIFICATION}"


def append_ignore_comment(line: str, codes: set[str]) -> str:
    """Append a justified suppression to a source line, preserving its newline."""
    ending = "\n" if line.endswith("\n") else ""
    body = line[:-1] if ending else line
    if _EXISTING_IGNORE.search(body):
        return line
    return f"{body.rstrip()}  {ignore_comment(codes)}{ending}"


def add_ignores(root: Path, selected: list[str] | None = None) -> dict:
    """Write suppression comments for active R* findings on Python call sites, then re-scan.

    Notebooks and non-suppressible findings (P*/S*) are left untouched. Comments use the
    same `# repro-lens: ignore[CODE] -- reason` form the scanner already understands.
    """
    root = root.resolve()
    report = check(root, selected)
    custom_rules = load_custom_rules(root)
    suppressible = {*SUPPRESSIBLE_CODES, *(r.id for r in custom_rules)}
    by_file: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for finding in report["findings"]:
        code = finding["code"]
        if code not in suppressible or "cell" in finding:
            continue
        by_file[finding["path"]][finding["line"]].add(code)
    for relative, lines in sorted(by_file.items()):
        path = root / relative
        if path.suffix != ".py" or not path.is_file():
            continue
        with tokenize.open(path) as handle:
            encoding = handle.encoding
            source_lines = handle.readlines()
        changed = False
        for lineno, codes in sorted(lines.items()):
            if not 1 <= lineno <= len(source_lines):
                continue
            updated = append_ignore_comment(source_lines[lineno - 1], codes)
            if updated != source_lines[lineno - 1]:
                source_lines[lineno - 1] = updated
                changed = True
        if changed:
            path.write_text("".join(source_lines), encoding=encoding)
    return check(root, selected)


def _paint(text: str, style: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{style}m{text}\033[0m"


def _visual_column(line: str, column: int, tabsize: int = 8) -> tuple[str, int]:
    """Expand tabs the way a terminal does, and map a 1-based source column."""
    visual: list[str] = []
    visual_col = 1
    mapped = None
    for index, char in enumerate(line, start=1):
        if index == column:
            mapped = visual_col
        if char == "\t":
            width = tabsize - (visual_col - 1) % tabsize
            visual.append(" " * width)
            visual_col += width
        else:
            visual.append(char)
            visual_col += 1
    return "".join(visual), mapped if mapped is not None else visual_col


def _source_line(root: Path, finding: dict) -> str | None:
    path = root / finding["path"]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if "cell" in finding:
        try:
            cell = json.loads(text)["cells"][finding["cell"] - 1]
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if not isinstance(source, str):
                return None
            lines = source.splitlines()
        except (json.JSONDecodeError, IndexError, KeyError, TypeError, AttributeError):
            return None
    else:
        lines = text.splitlines()
    index = finding["line"] - 1
    if not 0 <= index < len(lines):
        return None
    return lines[index]


def _location(finding: dict) -> str:
    if "cell" in finding:
        place = f"{finding['path']}:cell {finding['cell']}:{finding['line']}:{finding['column']}"
    else:
        place = f"{finding['path']}:{finding['line']}:{finding['column']}"
    return place


def _pretty_finding(root: Path, finding: dict, color: bool) -> list[str]:
    style = {"error": "1;31", "warning": "1;33", "review": "1;36"}.get(finding["severity"], "1")
    label = _paint(f"{finding['severity']}[{finding['code']}]", style, color)
    rows = [f"{label}: {finding['message']}", f"  --> {_location(finding)}"]
    source = _source_line(root, finding)
    if source is None:
        rows.append(f"   = help: {finding['suggestion']}")
        return rows
    visual, column = _visual_column(source, finding["column"])
    shown = visual.rstrip()
    start = min(max(column, 1) - 1, len(shown))
    end = max(len(shown), start + 1)
    carets = _paint("^" * (end - start), style, color)
    number = str(finding["line"])
    gutter = " " * len(number)
    rows.extend(
        [
            f"{gutter} |",
            f"{number} | {shown}",
            f"{gutter} | {' ' * start}{carets}",
            f"{gutter} |",
            f"{gutter} = help: {finding['suggestion']}",
        ]
    )
    return rows


def render_pretty(report: dict, *, color: bool = False) -> str:
    lines = [
        f"Repro Lens — {report['files_checked']} Python files and notebooks checked",
        report["assurance"],
        "",
    ]
    findings = report["findings"]
    root = Path(report["root"])
    if not findings:
        lines.append("No findings from the enabled checks.")
    for index, finding in enumerate(findings):
        if index:
            lines.append("")
        lines.extend(_pretty_finding(root, finding, color))
    if report["suppressed"]:
        if findings:
            lines.append("")
        lines.append(f"Suppressed findings: {len(report['suppressed'])}")
    return "\n".join(lines) + "\n"


def render(report: dict, format_: str, *, color: bool = False) -> str:
    if format_ == "json":
        return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if format_ == "pretty" and report.get("kind") == "static_check":
        return render_pretty(report, color=color)
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


def _md_cell(value: object) -> str:
    """Flatten a value for a GitHub-flavored Markdown table cell."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", " ").replace("|", "\\|").strip() or "—"


def _finding_location(finding: dict, prefix: str) -> str:
    path = f"{prefix.rstrip('/')}/{finding['path']}" if prefix else finding["path"]
    if "cell" in finding:
        return f"{path} (cell {finding['cell']}, line {finding['line']})"
    return f"{path}:{finding['line']}:{finding['column']}"


def _details(summary: str, body_lines: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", "", *body_lines, "", "</details>"]


def render_step_summary(report: dict, prefix: str = "") -> str:
    """Render a GitHub Actions job summary (`$GITHUB_STEP_SUMMARY`) for a report."""
    kind = report.get("kind")
    if kind == "static_check":
        findings = report.get("findings", [])
        lines = [
            "## Repro Lens",
            "",
            _md_cell(report.get("assurance", "")),
            "",
            f"Checked **{report.get('files_checked', 0)}** Python files and notebooks; "
            f"**{len(findings)}** finding{'s' if len(findings) != 1 else ''}.",
            "",
        ]
        if findings:
            lines += [
                "| Severity | Code | Location | Message |",
                "| --- | --- | --- | --- |",
            ]
            lines += [
                "| "
                + " | ".join(
                    [
                        _md_cell(finding["severity"]),
                        _md_cell(finding["code"]),
                        f"`{_md_cell(_finding_location(finding, prefix))}`",
                        _md_cell(finding["message"]),
                    ]
                )
                + " |"
                for finding in findings
            ]
            lines.append("")
            suggestions = [
                f"- **{finding['code']}** (`{_md_cell(_finding_location(finding, prefix))}`): "
                f"{_md_cell(finding['suggestion'])}"
                for finding in findings
            ]
            lines.extend(_details("Finding details", suggestions))
        else:
            lines.append("No findings from the enabled checks.")
        if report.get("suppressed"):
            lines += ["", f"Suppressed findings: {len(report['suppressed'])}"]
        return "\n".join(lines) + "\n"

    if kind in {"repeatability_test", "report_comparison"}:
        status = report.get("status", "unknown")
        lines = [
            f"## Repro Lens — `{_md_cell(status)}`",
            "",
            _md_cell(report.get("assurance", "")),
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | `{_md_cell(status)}` |",
        ]
        if kind == "repeatability_test" and report.get("report_path"):
            lines.append(f"| Evidence | `{_md_cell(report['report_path'])}` |")
        if kind == "report_comparison":
            for side in ("before", "after"):
                info = report.get(side) or {}
                if info.get("path"):
                    lines.append(f"| {side.title()} | `{_md_cell(info['path'])}` |")
        lines.append("")
        if report.get("error"):
            lines.extend(_details("Error", [f"```\n{report['error']}\n```"]))
            lines.append("")
        differences = list(report.get("differences") or [])
        if differences:
            body = [f"{index}. {_md_cell(item)}" for index, item in enumerate(differences, 1)]
            lines.extend(_details("Output differences", body))
        elif status == "matched":
            lines.append("No output differences.")
        elif status == "not_comparable":
            changes = []
            for category in ("policy_changes", "environment_changes"):
                for name in report.get(category) or {}:
                    changes.append(f"- {_md_cell(category)}: `{_md_cell(name)}`")
            for change, names in (report.get("input_changes") or {}).items():
                changes.extend(f"- Input {_md_cell(change)}: `{_md_cell(name)}`" for name in names)
            if changes:
                lines.extend(_details("Why reports are not comparable", changes))
            else:
                lines.append("Reports are not comparable.")
        return "\n".join(lines) + "\n"

    raise ValueError(f"Unsupported report kind for step summary: {kind!r}")


def write_step_summary(report: dict, prefix: str = "") -> None:
    """Append a job summary when running inside GitHub Actions."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write(render_step_summary(report, prefix))
