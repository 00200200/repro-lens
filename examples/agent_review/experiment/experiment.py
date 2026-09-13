"""Score a fixed synthetic dataset with a threshold classifier."""

import argparse
import csv
import json
from pathlib import Path

THRESHOLD = 0.5


def predict(score):
    return int(score >= THRESHOLD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with Path("samples.csv").open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))
    predictions = [predict(float(row["score"])) for row in samples]
    accuracy = sum(
        prediction == int(row["label"])
        for row, prediction in zip(samples, predictions, strict=True)
    ) / len(samples)
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "prediction"])
        writer.writerows(
            (row["sample_id"], prediction)
            for row, prediction in zip(samples, predictions, strict=True)
        )
    (args.output / "result.json").write_text(
        json.dumps({"metrics": {"accuracy": accuracy}}, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
