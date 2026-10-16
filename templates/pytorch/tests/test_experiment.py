from pathlib import Path

from ml_project.train import load_data, split_indices

ROOT = Path(__file__).resolve().parents[1]


def test_data_contract():
    features, labels = load_data(ROOT / "data/raw/synthetic.csv")
    assert len(features) == len(labels) == 240
    assert set(labels) == {0, 1}
    assert all(len(row) == 2 for row in features)


def test_split_indices_is_deterministic_and_disjoint():
    train_ids, test_ids = split_indices(240, 0.25, seed=1729)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(240))
    assert len(test_ids) == 60
    assert len(train_ids) == 180
    t1, te1 = split_indices(240, 0.25, seed=1729)
    assert train_ids == t1
    assert test_ids == te1


def test_run_experiment_contract(tmp_path):
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("PyTorch is not installed in the current environment")

    from ml_project.train import run_experiment

    result, train_ids, test_ids = run_experiment(ROOT / "configs/baseline.toml", tmp_path)
    assert set(train_ids).isdisjoint(test_ids)
    assert len(test_ids) == 60
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert result["metrics"]["loss"] >= 0
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "predictions.json").exists()
