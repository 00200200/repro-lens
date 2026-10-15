import json

from test_comparison import recorded_report, save

from repro_lens.capsule import create_capsule, inspect_capsule, verify_capsule


def test_capsule_is_deterministic_and_verifies_inputs(tmp_path):
    report = recorded_report()
    source = tmp_path / "train.py"
    source.write_text("print('ok')", encoding="utf-8")
    report["inputs_sha256"] = {
        "train.py": __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    }
    report_path = save(tmp_path / "report.json", report)
    first, second = tmp_path / "first.rlc", tmp_path / "second.rlc"
    create_capsule(report_path, first)
    create_capsule(report_path, second)
    assert first.read_bytes() == second.read_bytes()
    assert inspect_capsule(first)["capsule_version"] == 1
    assert verify_capsule(first, tmp_path)["status"] == "verified"


def test_capsule_reports_changed_and_missing_inputs(tmp_path):
    report = recorded_report()
    report["inputs_sha256"] = {"missing.csv": "a" * 64}
    report_path = save(tmp_path / "report.json", report)
    capsule = tmp_path / "capsule.rlc"
    create_capsule(report_path, capsule)
    result = verify_capsule(capsule, tmp_path)
    assert result["status"] == "drift"
    assert result["findings"] == [{"path": "missing.csv", "status": "missing"}]


def test_capsule_manifest_is_json_serializable(tmp_path):
    report_path = save(tmp_path / "report.json", recorded_report())
    capsule = tmp_path / "capsule.rlc"
    create_capsule(report_path, capsule)
    json.dumps(inspect_capsule(capsule))
