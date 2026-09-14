"""Render static check reports for code scanning (SARIF 2.1.0) and GitHub Actions annotations."""

from __future__ import annotations

import json

from . import __version__, frameworks
from .analysis import RULES

REPOSITORY = "https://github.com/00200200/repro-lens"
LEVELS = {"error": "error", "warning": "warning", "review": "note"}
COMMANDS = {"error": "error", "warning": "warning", "review": "notice"}


def help_uri(code: str) -> str:
    page = "frameworks.md" if code in frameworks.RULES else "rules.md"
    return f"{REPOSITORY}/blob/main/docs/{page}"


def located(finding: dict, prefix: str) -> tuple[str, str]:
    """Return the repository path and message; notebook lines are relative to a cell."""
    path = f"{prefix.rstrip('/')}/{finding['path']}" if prefix else finding["path"]
    text = f"{finding['message']} {finding['suggestion']}"
    if "cell" in finding:
        text = f"Cell {finding['cell']}, line {finding['line']}: {text}"
    return path, text


def to_sarif(report: dict, prefix: str = "") -> dict:
    codes = list(RULES)
    rules = [
        {
            "id": code,
            "shortDescription": {"text": RULES[code]},
            "helpUri": help_uri(code),
            "properties": {"tags": ["reproducibility"]},
        }
        for code in codes
    ]
    results = []
    for finding in report["findings"]:
        path, text = located(finding, prefix)
        notebook = "cell" in finding
        results.append(
            {
                "ruleId": finding["code"],
                "ruleIndex": codes.index(finding["code"]),
                "level": LEVELS[finding["severity"]],
                "message": {"text": text},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": path},
                            # A notebook's cell line does not map to a line of its JSON file.
                            "region": (
                                {"startLine": 1}
                                if notebook
                                else {
                                    "startLine": finding["line"],
                                    "startColumn": finding["column"],
                                }
                            ),
                        }
                    }
                ],
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Repro Lens",
                        "version": __version__,
                        "informationUri": REPOSITORY,
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def escape(value: str, property_: bool = False) -> str:
    value = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_:
        value = value.replace(":", "%3A").replace(",", "%2C")
    return value


def to_github(report: dict, prefix: str = "") -> str:
    lines = []
    for finding in report["findings"]:
        path, text = located(finding, prefix)
        properties = [f"file={escape(path, True)}"]
        if "cell" not in finding:
            properties += [f"line={finding['line']}", f"col={finding['column']}"]
        properties.append(f"title={escape('Repro Lens ' + finding['code'], True)}")
        command = COMMANDS[finding["severity"]]
        lines.append(f"::{command} {','.join(properties)}::{escape(text)}")
    count = len(report["findings"])
    lines.append(
        f"Repro Lens checked {report['files_checked']} Python files and notebooks: "
        f"{count} finding{'s' if count != 1 else ''}. {report['assurance']}"
    )
    return "\n".join(lines) + "\n"


def render_sarif(report: dict, prefix: str = "") -> str:
    return json.dumps(to_sarif(report, prefix), indent=2, ensure_ascii=False) + "\n"
