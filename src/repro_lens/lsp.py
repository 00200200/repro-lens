"""Small stdlib-only Language Server for live reproducibility diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from .analysis import analyze
from .project import check


def _send(message: dict) -> None:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
    sys.stdout.buffer.flush()


def _read() -> dict | None:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, _, value = line.decode("ascii").partition(":")
        headers[name.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    return json.loads(sys.stdin.buffer.read(length))


def _path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Unsupported document URI: {uri}")
    return Path(unquote(parsed.path.lstrip("/")))


def _diagnostics(findings: list[dict]) -> list[dict]:
    severity = {"error": 1, "warning": 2, "review": 3}
    return [
        {
            "range": {
                "start": {"line": max(f["line"] - 1, 0), "character": max(f["column"] - 1, 0)},
                "end": {"line": max(f["line"] - 1, 0), "character": max(f["column"], 1)},
            },
            "severity": severity.get(f["severity"], 3),
            "code": f["code"],
            "source": "repro-lens",
            "message": f"{f['message']} {f['suggestion']}",
        }
        for f in findings
    ]


def _publish(uri: str, findings: list[dict]) -> None:
    _send(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": uri, "diagnostics": _diagnostics(findings)},
        }
    )


def run(root: Path) -> None:
    open_documents: dict[str, str] = {}
    while True:
        message = _read()
        if message is None:
            return
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "capabilities": {
                            "textDocumentSync": {"openClose": True, "change": 1, "save": {}}
                        },
                        "serverInfo": {"name": "repro-lens"},
                    },
                }
            )
        elif method == "shutdown":
            _send({"jsonrpc": "2.0", "id": request_id, "result": None})
        elif method == "exit":
            return
        elif method == "textDocument/didOpen":
            document = message["params"]["textDocument"]
            uri, text = document["uri"], document["text"]
            open_documents[uri] = text
            relative = _path(uri).resolve().relative_to(root.resolve()).as_posix()
            findings, _ = analyze(text, relative)
            _publish(uri, [finding.to_dict() for finding in findings])
        elif method == "textDocument/didSave":
            uri = message["params"]["textDocument"]["uri"]
            report = check(root, [str(_path(uri))])
            _publish(uri, report["findings"])
        elif request_id is not None:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
