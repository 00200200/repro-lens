from __future__ import annotations

import argparse
import json
import keyword
import re
import shutil
import sys
from pathlib import Path

from . import __version__
from .analysis import RULES
from .comparison import compare_reports, render_comparison
from .project import check, render
from .verify import verify


def initialize(destination: Path, name: str):
    if (
        not re.fullmatch(r"[a-z][a-z0-9_]*", name)
        or keyword.iskeyword(name)
        or name in {"test", "tests", "src"}
    ):
        raise ValueError("Use a lowercase Python package name, e.g. iris_experiment")
    if destination.exists():
        raise ValueError(f"Destination already exists; nothing was overwritten: {destination}")
    packaged = Path(__file__).parent / "templates" / "sklearn"
    template = (
        packaged if packaged.is_dir() else Path(__file__).parents[2] / "templates" / "sklearn"
    )
    if not template.is_dir():
        raise ValueError("Template files are missing from this installation")
    shutil.copytree(
        template,
        destination,
        ignore=shutil.ignore_patterns(
            ".venv", "__pycache__", ".repro-lens", ".pytest_cache", ".ruff_cache"
        ),
    )
    for path in destination.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            content = content.replace("ml_project", name).replace(
                "ml-project", name.replace("_", "-")
            )
            path.write_text(content, encoding="utf-8")
    (destination / "src" / "ml_project").rename(destination / "src" / name)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducibility evidence for ML projects")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("check", help="Screen Python code without executing the project")
    scan.add_argument(
        "files", nargs="*", help="Selected project-relative paths; defaults to all Python files"
    )
    scan.add_argument("--root", type=Path, default=Path.cwd())
    scan.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    scan.add_argument("--output", type=Path)
    scan.add_argument("--fail-on", choices=["warning", "error"], default="warning")
    replay = sub.add_parser(
        "verify", help="Execute the configured local experiment twice (not sandboxed)"
    )
    replay.add_argument("--root", type=Path, default=Path.cwd())
    replay.add_argument("--format", choices=["text", "json"], default="text")
    comparison = sub.add_parser(
        "compare", help="Compare two retained verification reports without executing code"
    )
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    comparison.add_argument("--format", choices=["text", "json"], default="text")
    create = sub.add_parser(
        "init", help="Create a runnable scikit-learn project in a new directory"
    )
    create.add_argument("destination", type=Path)
    create.add_argument("--name", default="ml_project")
    sub.add_parser("rules", help="List supported rules")
    args = parser.parse_args(argv)
    try:
        if args.command == "rules":
            print("\n".join(f"{code} {description}" for code, description in RULES.items()))
            return 0
        if args.command == "init":
            destination = initialize(args.destination.resolve(), args.name)
            print(f"Created {destination}\nNext: cd {destination} && uv sync && uv run pytest")
            return 0
        if args.command == "compare":
            report = compare_reports(args.before, args.after)
            print(render_comparison(report, args.format), end="")
            return {"matched": 0, "mismatch": 1, "not_comparable": 2}[report["status"]]
        if not args.root.is_dir():
            raise ValueError(f"Project directory does not exist: {args.root}")
        if args.command == "verify":
            report = verify(args.root)
            print(render(report, args.format), end="")
            return {"matched": 0, "mismatch": 1, "error": 2}[report["status"]]
        report = check(args.root, args.files)
        output = render(report, args.format)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        failing = {"error"} | ({"warning"} if args.fail_on == "warning" else set())
        return int(any(f["severity"] in failing for f in report["findings"]))
    except (OSError, ValueError) as exc:
        if getattr(args, "format", None) == "json":
            print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        else:
            print(f"repro-lens: {exc}", file=sys.stderr)
        return 2
