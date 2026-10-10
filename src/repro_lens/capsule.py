"""Deterministic, local-only experiment evidence capsules."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from .comparison import read_report

CAPSULE_VERSION = 1
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(report: dict, report_hash: str) -> dict:
    artifacts = sorted(
        {
            name: digest
            for run in report["runs"]
            for name, digest in run["artifacts_sha256"].items()
        }.items()
    )
    return {
        "capsule_version": CAPSULE_VERSION,
        "report": {"path": REPORT_NAME, "sha256": report_hash},
        "git": report.get("git"),
        "environment": report["environment"],
        "configuration": report["configuration"],
        "inputs_sha256": dict(sorted(report["inputs_sha256"].items())),
        "artifacts_sha256": dict(artifacts),
        "external_artifacts": [],
    }


def create_capsule(report_path: Path, output_path: Path) -> Path:
    report, report_hash = read_report(report_path)
    manifest = _manifest(report, report_hash)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    report_bytes = report_path.read_bytes()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as capsule:
        for name, data in ((MANIFEST_NAME, manifest_bytes), (REPORT_NAME, report_bytes)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            capsule.writestr(info, data)
    return output_path


def read_capsule(path: Path) -> tuple[dict, bytes]:
    try:
        with zipfile.ZipFile(path) as capsule:
            names = set(capsule.namelist())
            if names != {MANIFEST_NAME, REPORT_NAME}:
                raise ValueError("Capsule must contain only manifest.json and report.json")
            manifest = json.loads(capsule.read(MANIFEST_NAME))
            report_bytes = capsule.read(REPORT_NAME)
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid capsule: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("capsule_version") != CAPSULE_VERSION:
        raise ValueError(f"Unsupported capsule version: {path}")
    if manifest.get("report", {}).get("sha256") != hashlib.sha256(report_bytes).hexdigest():
        raise ValueError("Capsule report hash does not match its manifest")
    return manifest, report_bytes


def inspect_capsule(path: Path) -> dict:
    manifest, _ = read_capsule(path)
    return manifest


def verify_capsule(path: Path, root: Path) -> dict:
    manifest, report_bytes = read_capsule(path)
    findings = []
    for name, expected in manifest["inputs_sha256"].items():
        candidate = root / name
        if not candidate.is_file():
            findings.append({"path": name, "status": "missing"})
        elif _sha256(candidate) != expected:
            findings.append({"path": name, "status": "changed"})
    return {
        "schema_version": 1,
        "kind": "capsule_verification",
        "status": "verified" if not findings else "drift",
        "capsule": str(path.resolve()),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "findings": findings,
    }
