from __future__ import annotations

import argparse
import json
import keyword
import os
import re
import shutil
import sys
from pathlib import Path

from . import __version__
from .analysis import RULES
from .capsule import create_capsule, inspect_capsule, verify_capsule
from .comparison import agent_review, compare_reports, render_agent_review, render_comparison
from .project import add_ignores, check, render, write_step_summary
from .sarif import render_sarif, to_github
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


def resolve_check_format(explicit: str | None, *, writing_file: bool, is_tty: bool) -> str:
    """Use pretty on a terminal. Pipes and --output stay on the line-oriented text report."""
    if explicit:
        return explicit
    if writing_file or not is_tty:
        return "text"
    return "pretty"


def ansi_enabled(*, writing_file: bool, is_tty: bool) -> bool:
    if writing_file or not is_tty:
        return False
    if "NO_COLOR" in os.environ or os.environ.get("FORCE_COLOR") == "0":
        return False
    return True


def repository_prefix(root: Path) -> str:
    """Code scanning resolves paths from the checkout, so prefix a root inside the cwd."""
    try:
        relative = root.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return ""
    return "" if relative == Path(".") else relative.as_posix()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducibility evidence for ML projects")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("check", help="Screen Python code without executing the project")
    scan.add_argument(
        "files", nargs="*", help="Selected project-relative paths; defaults to all Python files"
    )
    scan.add_argument("--root", type=Path, default=Path.cwd())
    scan.add_argument(
        "--format",
        choices=["text", "pretty", "json", "markdown", "sarif", "github"],
        help="Output style. Default: pretty on a terminal, text when piped or saved",
    )
    scan.add_argument("--output", type=Path)
    scan.add_argument("--fail-on", choices=["warning", "error"], default="warning")
    scan.add_argument(
        "--add-ignores",
        action="store_true",
        help=(
            "Append '# repro-lens: ignore[CODE] -- TODO: Review reproducibility' "
            "to each flagged Python call site"
        ),
    )
    replay = sub.add_parser(
        "verify", help="Execute the configured local experiment twice (not sandboxed)"
    )
    replay.add_argument("--root", type=Path, default=Path.cwd())
    replay.add_argument("--format", choices=["text", "json"], default="text")
    replay.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated integer seeds for multi-seed stability testing (e.g. 42,43,44)",
    )
    replay.add_argument(
        "--sandbox",
        choices=["docker", "podman"],
        default=None,
        help="Execute experiment in an isolated container sandbox (docker or podman)",
    )
    replay.add_argument(
        "--sandbox-image",
        type=str,
        default=None,
        help="Container image for sandbox execution (e.g. python:3.11-slim)",
    )
    comparison = sub.add_parser(
        "compare", help="Compare two retained verification reports without executing code"
    )
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    comparison.add_argument("--format", choices=["text", "json"], default="text")
    review = sub.add_parser(
        "agent-review", help="Review an agent-driven change against retained evidence"
    )
    review.add_argument("--baseline", required=True, type=Path)
    review.add_argument("--after", required=True, type=Path)
    review.add_argument("--format", choices=["text", "json"], default="json")
    capsule = sub.add_parser("capsule", help="Create or inspect a portable evidence capsule")
    capsule_sub = capsule.add_subparsers(dest="capsule_command", required=True)
    create_capsule_parser = capsule_sub.add_parser("create", help="Create a local evidence capsule")
    create_capsule_parser.add_argument("report", type=Path)
    create_capsule_parser.add_argument("output", type=Path)
    inspect_parser = capsule_sub.add_parser("inspect", help="Inspect capsule metadata")
    inspect_parser.add_argument("capsule", type=Path)
    verify_parser = capsule_sub.add_parser("verify", help="Check declared inputs against a capsule")
    verify_parser.add_argument("capsule", type=Path)
    verify_parser.add_argument("--root", type=Path, default=Path.cwd())
    for parser_ in (create_capsule_parser, inspect_parser, verify_parser):
        parser_.add_argument("--format", choices=["text", "json"], default="json")
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
            write_step_summary(report)
            return {"matched": 0, "mismatch": 1, "not_comparable": 2}[report["status"]]
        if args.command == "agent-review":
            review = agent_review(args.baseline, args.after)
            print(render_agent_review(review, args.format), end="")
            return {"APPROVED": 0, "REJECTED": 1, "NEEDS_HUMAN_REVIEW": 2}[review["verdict"]]
        if args.command == "capsule":
            if args.capsule_command == "create":
                result = {
                    "status": "created",
                    "capsule": str(create_capsule(args.report, args.output).resolve()),
                }
            elif args.capsule_command == "inspect":
                result = inspect_capsule(args.capsule)
            else:
                result = verify_capsule(args.capsule, args.root)
            if args.format == "json":
                print(json.dumps(result, indent=2, ensure_ascii=False) + "\n", end="")
            else:
                print(f"Repro Lens capsule: {result.get('status', 'inspected')}\n")
            return 0 if result.get("status") in {"created", "verified", None} else 1
        if not args.root.is_dir():
            raise ValueError(f"Project directory does not exist: {args.root}")
        if args.command == "verify":
            seeds = None
            if args.seeds is not None:
                try:
                    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
                except ValueError:
                    raise ValueError(
                        f"Invalid --seeds format: {args.seeds!r}. Expected integers."
                    ) from None
                if len(seeds) < 2:
                    raise ValueError("Multi-seed verification requires at least 2 seeds")
            report = verify(
                args.root,
                seeds=seeds,
                sandbox=args.sandbox,
                sandbox_image=args.sandbox_image,
            )
            print(render(report, args.format), end="")
            write_step_summary(report)
            exit_codes = {"matched": 0, "mismatch": 1, "stable": 0, "unstable": 1, "error": 2}
            return exit_codes.get(report["status"], 2)
        report = (
            add_ignores(args.root, args.files) if args.add_ignores else check(args.root, args.files)
        )
        writing_file = args.output is not None
        is_tty = sys.stdout.isatty()
        format_ = resolve_check_format(args.format, writing_file=writing_file, is_tty=is_tty)
        prefix = repository_prefix(args.root)
        if format_ in {"sarif", "github"}:
            renderer = render_sarif if format_ == "sarif" else to_github
            output = renderer(report, prefix)
        else:
            color = format_ == "pretty" and ansi_enabled(writing_file=writing_file, is_tty=is_tty)
            output = render(report, format_, color=color)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        write_step_summary(report, prefix)
        failing = {"error"} | ({"warning"} if args.fail_on == "warning" else set())
        return int(any(f["severity"] in failing for f in report["findings"]))
    except (OSError, ValueError) as exc:
        if getattr(args, "format", None) == "json":
            print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        else:
            print(f"repro-lens: {exc}", file=sys.stderr)
        return 2
