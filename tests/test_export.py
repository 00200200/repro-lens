import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repro_lens.cli import main
from repro_lens.verify import export_evidence


@pytest.fixture
def sample_report(tmp_path: Path) -> dict:
    report_file = tmp_path / "report.json"
    data = {
        "schema_version": 1,
        "kind": "repeatability_test",
        "status": "matched",
        "report_path": str(report_file),
        "statistics": {"accuracy": {"mean": 0.95, "std": 0.0, "min": 0.95, "max": 0.95}},
        "runs": [
            {
                "metrics": {"accuracy": 0.95},
                "artifacts_sha256": {"model.bin": "abcdef1234567890"},
            }
        ],
        "findings": [],
    }
    report_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def test_export_invalid_target(sample_report: dict):
    with pytest.raises(ValueError, match="Unsupported export target: 'tensorboard'"):
        export_evidence(sample_report, "tensorboard")


def test_export_wandb_not_installed(sample_report: dict):
    with patch.dict(sys.modules, {"wandb": None}):
        with pytest.raises(ValueError, match="Weights & Biases \\(wandb\\) is not installed"):
            export_evidence(sample_report, "wandb")


def test_export_mlflow_not_installed(sample_report: dict):
    with patch.dict(sys.modules, {"mlflow": None}):
        with pytest.raises(ValueError, match="MLflow is not installed"):
            export_evidence(sample_report, "mlflow")


def test_export_wandb_success(sample_report: dict):
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "wandb-run-123"
    mock_run.summary = {}
    mock_wandb.run = mock_run

    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        res = export_evidence(sample_report, "wandb")

    assert res["target"] == "wandb"
    assert res["status"] == "exported"
    assert res["run_id"] == "wandb-run-123"
    assert mock_run.summary["repro_lens_status"] == "matched"
    assert mock_run.summary["repro_lens_kind"] == "repeatability_test"
    assert mock_run.summary["repro_lens_accuracy_mean"] == 0.95
    assert mock_run.summary["repro_lens_run_1_accuracy"] == 0.95
    assert mock_run.summary["repro_lens_artifact_model.bin_sha256"] == "abcdef1234567890"
    mock_artifact.add_file.assert_called_once()
    mock_run.log_artifact.assert_called_once_with(mock_artifact)


def test_export_wandb_initializes_run_if_none(sample_report: dict):
    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_new_run = MagicMock()
    mock_new_run.id = "initialized-wandb-456"
    mock_new_run.summary = {}
    mock_wandb.init.return_value = mock_new_run

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        res = export_evidence(sample_report, "wandb")

    mock_wandb.init.assert_called_once_with(project="repro-lens", reinit=True)
    assert res["run_id"] == "initialized-wandb-456"


def test_export_mlflow_success(sample_report: dict):
    mock_mlflow = MagicMock()
    mock_active_run = MagicMock()
    mock_active_run.info.run_id = "mlflow-run-789"
    mock_mlflow.active_run.return_value = mock_active_run

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        res = export_evidence(sample_report, "mlflow")

    assert res["target"] == "mlflow"
    assert res["status"] == "exported"
    assert res["run_id"] == "mlflow-run-789"
    mock_mlflow.set_tag.assert_any_call("repro_lens_status", "matched")
    mock_mlflow.set_tag.assert_any_call("repro_lens_kind", "repeatability_test")
    mock_mlflow.set_tag.assert_any_call("repro_lens_artifact_model.bin_sha256", "abcdef1234567890")
    mock_mlflow.log_metric.assert_any_call("repro_lens_accuracy_mean", 0.95)
    mock_mlflow.log_metric.assert_any_call("repro_lens_run_1_accuracy", 0.95)
    mock_mlflow.log_dict.assert_called_once_with(sample_report, "repro_lens_report.json")


def test_export_mlflow_starts_run_if_none(sample_report: dict):
    mock_mlflow = MagicMock()
    mock_mlflow.active_run.return_value = None
    mock_new_run = MagicMock()
    mock_new_run.info.run_id = "started-mlflow-321"
    mock_mlflow.start_run.return_value = mock_new_run

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        res = export_evidence(sample_report, "mlflow")

    mock_mlflow.start_run.assert_called_once()
    assert res["run_id"] == "started-mlflow-321"


def test_cli_check_export_wandb(tmp_path: Path, capsys):
    (tmp_path / "model.py").write_text("print('clean')\n", encoding="utf-8")
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "cli-wandb-999"
    mock_run.summary = {}
    mock_wandb.run = mock_run

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        code = main(["check", "--root", str(tmp_path), "--export", "wandb"])

    assert code == 0
    captured = capsys.readouterr()
    assert "repro-lens: exported report to wandb (run: cli-wandb-999)" in captured.out
    assert mock_run.summary["repro_lens_findings_count"] == 0


def test_cli_compare_export_mlflow(tmp_path: Path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    run = {
        "returncode": 0,
        "metrics": {"score": 1},
        "runtime": {},
        "artifacts_sha256": {"pred.json": "a" * 64},
    }
    rep = {
        "schema_version": 1,
        "kind": "repeatability_test",
        "status": "matched",
        "differences": [],
        "configuration": {
            "command": ["python", "train.py", "{output}"],
            "inputs": ["train.py"],
            "metrics": ["score"],
            "artifacts": ["pred.json"],
            "timeout": 60,
            "atol": 0,
            "rtol": 0,
            "result": "result.json",
            "hash-inputs": True,
        },
        "environment": {"runner_python": "3.11", "platform": "test", "machine": "test"},
        "inputs_sha256": {"train.py": "b" * 64},
        "git": {"commit": None, "dirty": None},
        "runs": [run, run],
    }
    before.write_text(json.dumps(rep), encoding="utf-8")
    after.write_text(json.dumps(rep), encoding="utf-8")

    mock_mlflow = MagicMock()
    mock_active_run = MagicMock()
    mock_active_run.info.run_id = "cli-mlflow-555"
    mock_mlflow.active_run.return_value = mock_active_run

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        code = main(["compare", str(before), str(after), "--export", "mlflow"])

    assert code == 0
    captured = capsys.readouterr()
    assert "repro-lens: exported comparison report to mlflow (run: cli-mlflow-555)" in captured.out
    mock_mlflow.set_tag.assert_any_call("repro_lens_status", "matched")
