from pathlib import Path

from ml_project.train import load_data, run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_data_contract():
    features, labels = load_data(ROOT / "data/raw/synthetic.csv")
    assert len(features) == len(labels) == 240
    assert set(labels) == {0, 1}
    assert all(len(row) == 2 for row in features)


def test_split_is_disjoint_and_predictions_cover_holdout(tmp_path):
    result, train_ids, test_ids = run_experiment(ROOT / "configs/baseline.toml", tmp_path)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(240))
    assert len(test_ids) == 60
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert result["metrics"]["log_loss"] >= 0
