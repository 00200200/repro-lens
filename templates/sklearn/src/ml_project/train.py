from __future__ import annotations

import argparse
import csv
import json
import platform
import tomllib
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    features = [[float(row["x1"]), float(row["x2"])] for row in rows]
    labels = [int(row["target"]) for row in rows]
    return features, labels


def run_experiment(config_path: Path, output: Path):
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    settings = config["experiment"]
    seed = settings["seed"]
    if type(seed) is not int:
        raise ValueError("experiment.seed must be an integer")
    # Paths are relative to the project, not to the caller's home directory.
    root = config_path.resolve().parent.parent
    features, labels = load_data(root / settings["data"])
    train_ids, test_ids = train_test_split(
        list(range(len(labels))),
        test_size=settings["test_size"],
        stratify=labels,
        random_state=seed,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", random_state=seed, **config["model"]),
    )
    # Scaling is fitted only on the training partition through the pipeline.
    model.fit([features[i] for i in train_ids], [labels[i] for i in train_ids])
    probabilities = model.predict_proba([features[i] for i in test_ids])
    predicted = model.classes_[probabilities.argmax(axis=1)]
    truth = [labels[i] for i in test_ids]
    result = {
        "metrics": {
            "accuracy": float(accuracy_score(truth, predicted)),
            "log_loss": float(log_loss(truth, probabilities)),
        },
        "runtime": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "seed": seed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    predictions = {"test_ids": test_ids, "probabilities": probabilities.tolist()}
    (output / "predictions.json").write_text(
        json.dumps(predictions, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result, train_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    args = parser.parse_args()
    result, _, _ = run_experiment(args.config, args.output)
    print(json.dumps(result["metrics"]))


if __name__ == "__main__":
    main()
