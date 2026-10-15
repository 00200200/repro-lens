"""Create or update the Repro Lens summary comment for a pull request."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

MARKER = "<!-- repro-lens-summary -->"


def request(token: str, url: str, method: str = "GET", payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read()) if response.status != 204 else None


def render(report: dict) -> str:
    findings = report.get("findings", [])
    counts = {
        severity: sum(f.get("severity") == severity for f in findings)
        for severity in ("warning", "error")
    }
    lines = [
        MARKER,
        "### Repro Lens Check Summary",
        "",
        f"- **Files checked:** {report.get('files_checked', 0)}",
    ]
    lines.append(f"- **Findings:** {counts['warning']} warnings, {counts['error']} errors")
    if findings:
        lines.extend(["", "| Rule | File | Line | Message |", "| --- | --- | ---: | --- |"])
        for finding in findings:
            path = str(finding.get("path", "")).replace("|", "\\|")
            message = str(finding.get("message", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {finding.get('code', '')} | {path} | {finding.get('line', '')} | {message} |"
            )
    else:
        lines.extend(["", "No reproducibility findings were reported."])
    return "\n".join(lines) + "\n"


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    report_path = os.environ.get("REPRO_LENS_REPORT")
    if not all((token, repository, event_path, report_path)):
        return 0
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    number = event.get("pull_request", {}).get("number")
    if not number:
        return 0
    api = f"https://api.github.com/repos/{repository}/issues/{number}/comments"
    body = render(json.loads(Path(report_path).read_text(encoding="utf-8")))
    comments = request(token, api)
    existing = next((item for item in comments if MARKER in item.get("body", "")), None)
    if existing:
        request(token, f"{api}/{existing['id']}", method="PATCH", payload={"body": body})
    else:
        request(token, api, method="POST", payload={"body": body})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
