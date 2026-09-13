"""Fit a small synthetic regression fixture on one CPU thread."""

import argparse
import csv
import json
import platform
import tomllib
from pathlib import Path

import numpy as np
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))
    values = np.loadtxt("samples.csv", delimiter=",", skiprows=1, dtype=np.float32)
    features = values[:, :2]
    target = values[:, 2]
    data = xgb.DMatrix(features, label=target, nthread=1)
    model = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cpu",
            "max_depth": config["max_depth"],
            "seed": config["seed"],
            "nthread": 1,
        },
        data,
        num_boost_round=config["rounds"],
    )
    predictions = model.predict(data)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "prediction"])
        writer.writerows(enumerate(float(value) for value in predictions))
    result = {
        "metrics": {"training_rmse": float(np.sqrt(np.mean((predictions - target) ** 2)))},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
            "device": "cpu",
            "threads": 1,
        },
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
