import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_pyodide", ROOT / "tools/fetch_pyodide.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def archive(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


RUNTIME = {f"package/{name}": name.encode() for name in fetch.FILES}


def test_extracts_only_the_runtime_files(tmp_path):
    data = archive({**RUNTIME, "package/console.html": b"<html>", "../escape.js": b"no"})
    names = fetch.extract_runtime(data, fetch.integrity(data), tmp_path / "pyodide")
    assert names == list(fetch.FILES)
    assert sorted(path.name for path in (tmp_path / "pyodide").iterdir()) == sorted(fetch.FILES)
    assert (tmp_path / "pyodide/pyodide.mjs").read_bytes() == b"pyodide.mjs"
    assert not (tmp_path / "escape.js").exists()


def test_rejects_an_archive_that_does_not_match_the_pin(tmp_path):
    data = archive(RUNTIME)
    with pytest.raises(ValueError, match="integrity"):
        fetch.extract_runtime(data, fetch.integrity(data + b"x"), tmp_path / "pyodide")
    assert not (tmp_path / "pyodide").exists()


def test_rejects_an_archive_missing_runtime_files(tmp_path):
    files = dict(RUNTIME)
    del files["package/python_stdlib.zip"]
    data = archive(files)
    with pytest.raises(ValueError, match="python_stdlib.zip"):
        fetch.extract_runtime(data, fetch.integrity(data), tmp_path / "pyodide")
    assert not (tmp_path / "pyodide").exists()


def test_pin_is_a_complete_sha512_integrity_value():
    assert fetch.INTEGRITY.startswith("sha512-") and len(fetch.INTEGRITY) == 95
    assert fetch.URL.endswith(f"pyodide-{fetch.VERSION}.tgz")
