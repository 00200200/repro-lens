from __future__ import annotations

import ast
import concurrent.futures
import difflib
import fnmatch
import importlib.metadata
import json
import os
import re
import subprocess
import tokenize
import tomllib
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from .analysis import RULES, CustomRule, Finding, analyze
from .frameworks import RULES as FRAMEWORK_RULES
from .notebooks import check_execution_order, notebook_source

LOCKFILE_NAMES = ("uv.lock", "poetry.lock")
# Rule codes a suppression comment may list.
SUPPRESSIBLE_CODES = {"R101", "R102", "R103", "R131", "R190", *FRAMEWORK_RULES}
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
    allowed = {"exclude", "required-files", "verify", "ignore", "severities", "baseline"}
    unknown = set(policy) - allowed
    if unknown:
        raise ValueError(f"Unknown repro-lens settings: {sorted(unknown)}")
    for field in ("exclude", "required-files"):
        value = policy.get(field, [])
        if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
            raise ValueError(f"{field} must be a list of nonempty strings")

    if "baseline" in policy:
        baseline_val = policy["baseline"]
        if not isinstance(baseline_val, str) or not baseline_val.strip():
            raise ValueError("baseline must be a nonempty string path")

    known_rules = set(RULES)
    for custom in load_custom_rules(root):
        known_rules.add(custom.id)

    if "ignore" in policy:
        ignore_rules = policy["ignore"]
        if not isinstance(ignore_rules, list) or any(
            not isinstance(x, str) or not x for x in ignore_rules
        ):
            raise ValueError("ignore must be a list of nonempty strings")
        for rule_id in ignore_rules:
            if rule_id not in known_rules:
                raise ValueError(f"Unknown rule code in ignore: {rule_id}")

    if "severities" in policy:
        severities = policy["severities"]
        if not isinstance(severities, dict):
            raise ValueError("severities must be a table")
        valid_severities = {"warning", "review", "error"}
        for rule_id, sev in severities.items():
            if not isinstance(rule_id, str) or rule_id not in known_rules:
                raise ValueError(f"Unknown rule code in severities: {rule_id}")
            if not isinstance(sev, str) or sev not in valid_severities:
                expected = sorted(valid_severities)
                raise ValueError(
                    f"Invalid severity for {rule_id}: {sev!r}. Must be one of {expected}"
                )
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
    raw = path.read_text(encoding="utf-8")
    source, locations, invalid = notebook_source(raw)
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
    order_active, order_ignored = check_execution_order(raw, relative)
    return [in_cell(f) for f in active] + errors + order_active, [
        in_cell(f) for f in ignored
    ] + order_ignored


def _scan_path(args):
    path, relative, custom_rules = args
    try:
        if path.suffix == ".ipynb":
            return analyze_notebook(path, relative, custom_rules=custom_rules)
        with tokenize.open(path) as handle:
            return analyze(handle.read(), relative, custom_rules=custom_rules)
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        return [
            Finding(relative, 1, 1, "S902", "error", str(exc), "Restore readable Python source.")
        ], []


def read_baseline(path: Path) -> list[dict]:
    """Read a .repro-lens-baseline.json file and return list of recorded findings."""
    if not path.is_file():
        raise ValueError(f"Baseline file does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict) or "findings" not in data:
        raise ValueError(f"Baseline file must contain a 'findings' list: {path}")
    findings = data["findings"]
    if not isinstance(findings, list):
        raise ValueError(f"Baseline 'findings' must be a list: {path}")
    return findings


def write_baseline(path: Path, findings: list[dict]) -> None:
    """Write detected findings to a portable baseline JSON file."""
    portable_findings = [
        {
            "path": Path(f["path"]).as_posix(),
            "code": f["code"],
            "line": f.get("line", 1),
            "column": f.get("column", 1),
            "message": f.get("message", ""),
        }
        for f in findings
    ]
    portable_findings.sort(key=lambda f: (f["path"], f["line"], f["column"], f["code"]))
    payload = {
        "schema_version": 1,
        "generator": "repro-lens",
        "findings": portable_findings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_baseline(report: dict, baseline_findings: list[dict]) -> dict:
    """Filter findings present in baseline from report['findings'] into report['suppressed']."""
    baseline_exact = set((f["path"], f["code"], f["line"], f["column"]) for f in baseline_findings)
    baseline_counts = Counter((f["path"], f["code"]) for f in baseline_findings)

    unmatched: list[dict] = []
    suppressed: list[dict] = list(report.get("suppressed", []))

    remaining_report: list[dict] = []
    for f in report.get("findings", []):
        key_exact = (f["path"], f["code"], f["line"], f["column"])
        key_code = (f["path"], f["code"])
        if key_exact in baseline_exact and baseline_counts[key_code] > 0:
            baseline_exact.remove(key_exact)
            baseline_counts[key_code] -= 1
            suppressed.append(f)
        else:
            remaining_report.append(f)

    for f in remaining_report:
        key_code = (f["path"], f["code"])
        if baseline_counts[key_code] > 0:
            baseline_counts[key_code] -= 1
            suppressed.append(f)
        else:
            unmatched.append(f)

    stale_count = sum(baseline_counts.values())
    new_report = dict(report)
    new_report["findings"] = unmatched
    new_report["suppressed"] = suppressed
    if stale_count > 0:
        new_report["stale_baseline_entries"] = stale_count
    return new_report


def check(
    root: Path,
    selected: list[str] | None = None,
    jobs: int = 1,
    baseline: Path | str | None = None,
) -> dict:
    root = root.resolve()
    if jobs < 1:
        raise ValueError("jobs must be a positive integer")
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
    paths = list(paths_to_scan(root, selected or [], policy.get("exclude", [])))
    count = len(paths)
    tasks = [(path, relative, tuple(custom_rules)) for path, relative in paths]
    if jobs > 1 and count > 20:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            results = executor.map(_scan_path, tasks)
    else:
        results = map(_scan_path, tasks)
    for active, ignored in results:
        findings.extend(active)
        suppressed.extend(ignored)
    ignored_rule_codes = set(policy.get("ignore", []))
    severity_overrides = policy.get("severities", {})
    if ignored_rule_codes or severity_overrides:
        filtered_findings = []
        for f in findings:
            if f.code in ignored_rule_codes:
                continue
            if f.code in severity_overrides:
                f = replace(f, severity=severity_overrides[f.code])
            filtered_findings.append(f)
        findings = filtered_findings

        filtered_suppressed = []
        for f in suppressed:
            if f.code in ignored_rule_codes:
                continue
            if f.code in severity_overrides:
                f = replace(f, severity=severity_overrides[f.code])
            filtered_suppressed.append(f)
        suppressed = filtered_suppressed
    findings.sort(key=lambda f: (f.path, f.cell or 0, f.line, f.code))
    report = {
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
    baseline_target = baseline if baseline is not None else policy.get("baseline")
    if baseline_target is not None:
        baseline_path = (
            Path(baseline_target) if Path(baseline_target).is_absolute() else root / baseline_target
        )
        baseline_findings = read_baseline(baseline_path)
        report = apply_baseline(report, baseline_findings)
    return report


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


SEED_KWARG_RULES = {
    "R101": "random_state",
    "R115": "random_state",
    "R129": "random_state",
    "R118": "seed",
    "R121": "seed",
    "R127": "seed",
}


def find_call_node(tree: ast.AST, line: int, col: int) -> ast.Call | None:
    """Locate the ast.Call node matching or containing the given 1-indexed line and col."""
    target_col = col - 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.lineno == line and node.col_offset == target_col:
            return node
    line_calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and node.lineno == line
    ]
    if line_calls:
        line_calls.sort(key=lambda n: abs(n.col_offset - target_col))
        return line_calls[0]
    span_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.lineno <= line <= getattr(node, "end_lineno", node.lineno)
    ]
    if span_calls:
        span_calls.sort(
            key=lambda n: (
                getattr(n, "end_lineno", n.lineno) - n.lineno,
                abs(n.col_offset - target_col),
            )
        )
        return span_calls[0]
    return None


def insert_call_argument(source: str, call: ast.Call, kwarg_str: str) -> str:
    """Insert or update a keyword argument into an ast.Call, preserving layout."""
    lines = source.splitlines(keepends=True)
    end_line_idx = call.end_lineno - 1
    end_col = call.end_col_offset - 1

    last_arg = None
    if call.keywords:
        last_arg = call.keywords[-1]
    elif call.args:
        last_arg = call.args[-1]

    if last_arg is None:
        if call.lineno == call.end_lineno:
            line = lines[end_line_idx]
            lines[end_line_idx] = line[:end_col] + kwarg_str + line[end_col:]
        else:
            call_start_line = lines[call.lineno - 1]
            call_indent = call_start_line[: len(call_start_line) - len(call_start_line.lstrip())]
            arg_indent = call_indent + "    "
            lines.insert(end_line_idx, f"{arg_indent}{kwarg_str},\n")
    else:
        if last_arg.end_lineno == call.end_lineno:
            line = lines[end_line_idx]
            before_paren = line[:end_col].rstrip()
            if before_paren.endswith(","):
                new_line = before_paren + " " + kwarg_str + line[end_col:]
            else:
                new_line = before_paren + ", " + kwarg_str + line[end_col:]
            lines[end_line_idx] = new_line
        else:
            last_line = lines[last_arg.end_lineno - 1]
            after_last = last_line[last_arg.end_col_offset :]
            if not after_last.lstrip().startswith(","):
                lines[last_arg.end_lineno - 1] = (
                    last_line[: last_arg.end_col_offset] + "," + after_last
                )

            closing_line = lines[end_line_idx]
            before_paren = closing_line[:end_col]
            if before_paren.strip() == "":
                arg_line = lines[last_arg.end_lineno - 1]
                arg_indent = arg_line[: len(arg_line) - len(arg_line.lstrip())]
                lines.insert(end_line_idx, f"{arg_indent}{kwarg_str},\n")
            else:
                before_paren_s = before_paren.rstrip()
                if before_paren_s.endswith(","):
                    new_line = before_paren_s + " " + kwarg_str + closing_line[end_col:]
                else:
                    new_line = before_paren_s + ", " + kwarg_str + closing_line[end_col:]
                lines[end_line_idx] = new_line

    return "".join(lines)


def resolve_fix_kwarg(
    finding: dict,
    node: ast.Call | None,
    custom_rules: dict[str, CustomRule],
    seed: int = 42,
) -> str | None:
    code = finding.get("code")
    if code in SEED_KWARG_RULES:
        kw = SEED_KWARG_RULES[code]
        return f"{kw}={seed}"
    if code == "R102":
        if node and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "SeedSequence")
            or (isinstance(node.func, ast.Name) and node.func.id == "SeedSequence")
        ):
            return f"entropy={seed}"
        return f"seed={seed}"
    if code == "R103":
        return f"x={seed}"
    if code in custom_rules:
        crule = custom_rules[code]
        if crule.missing_kwargs:
            return f"{crule.missing_kwargs[0]}={seed}"
    return None


def suggest_project_fixes(
    root: Path,
    selected: list[str] | None = None,
    seed: int = 42,
    ignores_only: bool = False,
) -> tuple[dict[str, str], dict]:
    """Compute automated fixes for findings in Python source files.

    Returns (modifications, report) where modifications maps relative path to new source.
    """
    root = root.resolve()
    report = check(root, selected)
    custom_rules = {r.id: r for r in load_custom_rules(root)}
    suppressible = {*SUPPRESSIBLE_CODES, *custom_rules.keys()}

    by_file: dict[str, list[dict]] = defaultdict(list)
    for finding in report["findings"]:
        if "cell" in finding:
            continue
        code = finding["code"]
        if ignores_only:
            if code in suppressible:
                by_file[finding["path"]].append(finding)
        else:
            if code in SEED_KWARG_RULES or code in {"R102", "R103"} or code in suppressible:
                by_file[finding["path"]].append(finding)

    modifications: dict[str, str] = {}
    for relative, findings in sorted(by_file.items()):
        path = root / relative
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            with tokenize.open(path) as handle:
                original_source = handle.read()
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        current_source = original_source
        visited_findings: set[tuple[int, int, str]] = set()

        max_passes = len(findings) * 3 + 10
        for _ in range(max_passes):
            try:
                tree = ast.parse(current_source)
            except SyntaxError:
                break

            current_findings, _ = analyze(current_source, relative, list(custom_rules.values()))
            actionable = [
                f.to_dict() if hasattr(f, "to_dict") else f
                for f in current_findings
                if (f.line, f.column, f.code) not in visited_findings
                and (
                    f.code in suppressible
                    or f.code in SEED_KWARG_RULES
                    or f.code in {"R102", "R103"}
                )
            ]
            if not actionable:
                break

            target = max(actionable, key=lambda f: (f["line"], f["column"]))
            key = (target["line"], target["column"], target["code"])
            visited_findings.add(key)

            new_source = None
            if not ignores_only:
                call_node = find_call_node(tree, target["line"], target["column"])
                kwarg = resolve_fix_kwarg(target, call_node, custom_rules, seed=seed)
                if call_node and kwarg:
                    kw_name, kw_val = kwarg.split("=", 1)
                    existing_kw = next((k for k in call_node.keywords if k.arg == kw_name), None)
                    if existing_kw and existing_kw.value:
                        lines = current_source.splitlines(keepends=True)
                        val_node = existing_kw.value
                        if val_node.lineno == val_node.end_lineno:
                            l_idx = val_node.lineno - 1
                            line = lines[l_idx]
                            prefix = line[: val_node.col_offset]
                            suffix = line[val_node.end_col_offset :]
                            lines[l_idx] = f"{prefix}{kw_val}{suffix}"
                            new_source = "".join(lines)
                    else:
                        new_source = insert_call_argument(current_source, call_node, kwarg)

            if new_source is None and target["code"] in suppressible:
                lines = current_source.splitlines(keepends=True)
                if 1 <= target["line"] <= len(lines):
                    updated_line = append_ignore_comment(
                        lines[target["line"] - 1], {target["code"]}
                    )
                    if updated_line != lines[target["line"] - 1]:
                        lines[target["line"] - 1] = updated_line
                        new_source = "".join(lines)

            if new_source is not None and new_source != current_source:
                current_source = new_source

        if current_source != original_source:
            modifications[relative] = current_source

    return modifications, report


def render_diff(root: Path, modifications: dict[str, str]) -> str:
    """Generate a unified diff across all modified files."""
    root = root.resolve()
    diff_chunks: list[str] = []
    for relative, new_source in sorted(modifications.items()):
        path = root / relative
        if not path.is_file():
            continue
        try:
            with tokenize.open(path) as handle:
                orig_lines = handle.readlines()
        except OSError:
            orig_lines = []
        new_lines = new_source.splitlines(keepends=True)
        rel_posix = Path(relative).as_posix()
        diff = difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{rel_posix}",
            tofile=f"b/{rel_posix}",
        )
        diff_chunks.extend(diff)
    return "".join(diff_chunks)


def apply_project_fixes(root: Path, modifications: dict[str, str]) -> None:
    """Write modified source code back to disk using each file's encoding."""
    root = root.resolve()
    for relative, new_source in modifications.items():
        path = root / relative
        encoding = "utf-8"
        if path.is_file():
            try:
                with tokenize.open(path) as handle:
                    encoding = handle.encoding
            except OSError:
                pass
        path.write_text(new_source, encoding=encoding)


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
        if report.get("kind") == "multi_seed_verification":
            for metric, stats in (report.get("statistics") or {}).items():
                lines.append(
                    f"Metric '{metric}' (n={stats['count']}): mean={stats['mean']} "
                    f"std={stats['std']} variance={stats['variance']} "
                    f"min={stats['min']} max={stats['max']}"
                )
            for fail in report.get("stability_failures") or []:
                lines.append(f"Stability failure: {fail}")
        if report.get("sandbox"):
            lines.append(f"Sandbox: {report['sandbox']} ({report.get('sandbox_image', 'default')})")
        for warning in report.get("warnings", []):
            lines.append(f"Warning: {warning}")
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

    if kind in {"repeatability_test", "multi_seed_verification", "report_comparison"}:
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
        if report.get("sandbox"):
            lines.append(
                f"| Sandbox | `{_md_cell(report['sandbox'])}` "
                f"(`{_md_cell(report.get('sandbox_image', 'default'))}`) |"
            )
        if kind == "multi_seed_verification":
            seeds_str = ", ".join(str(s) for s in report.get("seeds", []))
            lines.append(f"| Seeds | `{_md_cell(seeds_str)}` |")
            stats = report.get("statistics", {})
            if stats:
                lines += [
                    "",
                    "### Metric Stability Across Seeds",
                    "",
                    "| Metric | Mean | Std Dev | Variance | Min | Max |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
                for name, s in stats.items():
                    lines.append(
                        f"| `{_md_cell(name)}` | {s['mean']} | {s['std']} | "
                        f"{s['variance']} | {s['min']} | {s['max']} |"
                    )
            failures = report.get("stability_failures", [])
            if failures:
                lines.extend(_details("Stability Failures", [f"- {_md_cell(f)}" for f in failures]))
            else:
                lines.append("")
                lines.append("All metrics satisfied stability bounds.")
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
        warnings = list(report.get("warnings") or [])
        if warnings:
            body = [f"- {_md_cell(item)}" for item in warnings]
            lines.extend(_details("Warnings", body))
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
