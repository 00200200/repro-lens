"""Fit a small synthetic classifier. XOR needs more than a depth-1 stump."""

import argparse
import csv
import json
import platform
import tomllib
from pathlib import Path

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))
    values = np.loadtxt("samples.csv", delimiter=",", skiprows=1)
    features = values[:, :2]
    labels = values[:, 2].astype(int)
    model = DecisionTreeClassifier(max_depth=config["max_depth"], random_state=config["seed"])
    model.fit(features, labels)
    predictions = model.predict(features)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "prediction"])
        writer.writerows(enumerate(int(value) for value in predictions))
    result = {
        "metrics": {"training_accuracy": float(np.mean(predictions == labels))},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
