import io
import json

from repro_lens import lsp


def frame(message):
    payload = json.dumps(message).encode()
    return f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload


def test_lsp_initialize_open_and_shutdown(tmp_path, monkeypatch):
    source = tmp_path / "train.py"
    uri = source.as_uri()
    messages = (
        frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        + frame(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {"uri": uri, "text": "import random\nrandom.Random()\n"}
                },
            }
        )
        + frame({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
        + frame({"jsonrpc": "2.0", "method": "exit", "params": {}})
    )
    output = io.BytesIO()
    monkeypatch.setattr(lsp.sys, "stdin", io.TextIOWrapper(io.BytesIO(messages)))
    monkeypatch.setattr(lsp.sys, "stdout", io.TextIOWrapper(output))
    lsp.run(tmp_path)
    raw = output.getvalue()
    assert b'"id":1' in raw
    assert b"textDocument/publishDiagnostics" in raw
    assert b'"code":"R103"' in raw
    assert b'"id":2' in raw
