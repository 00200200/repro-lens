"""Add a pinned, integrity-checked Pyodide runtime to the built gallery.

The live checker runs the Repro Lens analyzer in the visitor's browser. Pyodide is
served from the gallery itself, so the page loads no third-party scripts. Only the
runtime files needed for standard-library Python are extracted from the npm package.
"""

import argparse
import base64
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "314.0.6"
URL = f"https://registry.npmjs.org/pyodide/-/pyodide-{VERSION}.tgz"
# npm "dist.integrity" for this exact tarball.
INTEGRITY = (
    "sha512-BKDTJyIqFxC4BExLqeRS3f5xvXZIjOt8C3zGLN/"
    "Cc7tFxSwvKVhVkchQQ2AGLOtR4YrVOIFxbV8poyDOOmWwxQ=="
)
FILES = (
    "pyodide.mjs",
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)


def integrity(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()


def extract_runtime(archive: bytes, expected: str, output: Path) -> list[str]:
    """Verify the whole archive, then write only the known runtime files."""
    if integrity(archive) != expected:
        raise ValueError("Pyodide archive does not match the pinned integrity hash")
    wanted = {f"package/{name}": name for name in FILES}
    found = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            if member.name in wanted and member.isfile():
                found[wanted[member.name]] = bundle.extractfile(member).read()
    missing = sorted(set(FILES) - set(found))
    if missing:
        raise ValueError(f"Pyodide archive is missing: {', '.join(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        (output / name).write_bytes(found[name])
    return list(FILES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "dist/gallery/pyodide")
    parser.add_argument("--archive", type=Path, help="use a downloaded pyodide-*.tgz")
    args = parser.parse_args()
    if args.archive:
        archive = args.archive.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as response:
            archive = response.read()
    try:
        names = extract_runtime(archive, INTEGRITY, args.output)
    except (ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    size = sum((args.output / name).stat().st_size for name in names)
    print(f"Added Pyodide {VERSION} ({size / 1_000_000:.1f} MB) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
