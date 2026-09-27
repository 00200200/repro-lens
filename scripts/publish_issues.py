#!/usr/bin/env python3
"""Publish curated contribution issues to GitHub repository via `gh issue create`.

Usage:
    python scripts/publish_issues.py --list
    python scripts/publish_issues.py --dry-run
    python scripts/publish_issues.py --issue 1
    python scripts/publish_issues.py --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_issues() -> list[dict]:
    path = Path(__file__).parent / "issues.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_issue(issue: dict, repo: str, total: int = 40, dry_run: bool = False):
    print(f"\n[{issue['id']}/{total}] {issue['title']}")
    print(f"Labels: {', '.join(issue['labels'])}")
    if dry_run:
        print("(DRY RUN - skipping actual creation)")
        return
    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        issue["title"],
        "--body",
        issue["body"],
    ]
    for label in issue["labels"]:
        cmd.extend(["--label", label])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"Created: {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Publish curated contribution issues to GitHub.")
    parser.add_argument(
        "--repo",
        default="00200200/repro-lens",
        help="GitHub repo (default: 00200200/repro-lens)",
    )
    parser.add_argument("--list", action="store_true", help="List all issues")
    parser.add_argument("--dry-run", action="store_true", help="Print command without creating")
    parser.add_argument("--issue", type=int, help="Create a specific issue by ID (1-15)")
    parser.add_argument("--all", action="store_true", help="Create all issues on GitHub")
    args = parser.parse_args()

    issues = load_issues()

    if args.list:
        print(f"Available issues ({len(issues)}):")
        for iss in issues:
            labels = ", ".join(iss["labels"])
            print(f"  #{iss['id']:2d}: {iss['title']} [{labels}]")
        return

    if args.issue:
        matching = [iss for iss in issues if iss["id"] == args.issue]
        if not matching:
            print(f"Error: Issue #{args.issue} not found.")
            sys.exit(1)
        create_issue(matching[0], args.repo, total=len(issues), dry_run=args.dry_run)
        return

    if args.all:
        for iss in issues:
            create_issue(iss, args.repo, total=len(issues), dry_run=args.dry_run)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
