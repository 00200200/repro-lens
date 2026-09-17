"""The sklearn replay example must advertise a real compare, not only a static scan."""

from pathlib import Path

from repro_lens.analysis import analyze

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/sklearn_review"
EXPERIMENT = EXAMPLE / "experiment"


def test_sklearn_replay_fixture_is_explicit_and_scanned_clean():
    source = (EXPERIMENT / "train.py").read_text(encoding="utf-8")
    assert "DecisionTreeClassifier" in source
    assert "random_state" in source
    active, suppressed = analyze(source, "train.py")
    assert active == []
    assert suppressed == []


def test_sklearn_replay_demo_wires_an_equivalent_edit_and_a_mismatch():
    train = (EXPERIMENT / "train.py").read_text(encoding="utf-8")
    config = (EXPERIMENT / "config.toml").read_text(encoding="utf-8")
    demo = (EXAMPLE / "demo.py").read_text(encoding="utf-8")
    policy = (EXPERIMENT / "pyproject.toml").read_text(encoding="utf-8")

    assert "features = values[:, :2]" in train
    assert "values[:, [0, 1]]" in demo
    assert "max_depth = 3" in config
    assert "max_depth = 1" in demo
    assert '("refactor", "matched", ["train.py"])' in demo
    assert '("shallower tree", "mismatch", ["config.toml"])' in demo
    assert 'metrics = ["training_accuracy"]' in policy
    assert 'artifacts = ["predictions.csv"]' in policy
    assert (EXPERIMENT / "samples.csv").read_text(encoding="utf-8").count("\n") >= 16
    assert (EXPERIMENT / "uv.lock").is_file()
