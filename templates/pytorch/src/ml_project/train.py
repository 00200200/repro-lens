from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set CUBLAS workspace determinism configuration before CUDA initialization (satisfies R120).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_everything(seed: int) -> None:
    """Pin pseudo-random number generators and deterministic backend flags."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed per-worker random number generators for DataLoader multi-processing (satisfies R116)."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SyntheticDataset(Dataset):
    """Small in-memory PyTorch dataset for binary classification."""

    def __init__(self, features: list[list[float]], labels: list[int]) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class Classifier(nn.Module):
    """Small 2-layer classifier."""

    def __init__(self, in_features: int = 2, hidden_dim: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data(path: Path) -> tuple[list[list[float]], list[int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    features = [[float(row["x1"]), float(row["x2"])] for row in rows]
    labels = [int(row["target"]) for row in rows]
    return features, labels


def split_indices(n: int, test_size: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split_at = int(n * (1.0 - test_size))
    return sorted(indices[:split_at]), sorted(indices[split_at:])


def run_experiment(config_path: Path, output: Path) -> tuple[dict, list[int], list[int]]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    settings = config["experiment"]
    seed = settings["seed"]
    if not isinstance(seed, int):
        raise ValueError("experiment.seed must be an integer")

    seed_everything(seed)

    root = config_path.resolve().parent.parent
    features, labels = load_data(root / settings["data"])
    train_ids, test_ids = split_indices(len(labels), settings["test_size"], seed)

    train_features = [features[i] for i in train_ids]
    train_labels = [labels[i] for i in train_ids]
    test_features = [features[i] for i in test_ids]
    test_labels = [labels[i] for i in test_ids]

    train_dataset = SyntheticDataset(train_features, train_labels)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.get("batch_size", 32),
        shuffle=True,
        generator=g,
        num_workers=0,
        worker_init_fn=seed_worker,
    )

    model = Classifier(
        in_features=2,
        hidden_dim=config.get("model", {}).get("hidden_dim", 8),
        num_classes=2,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.get("lr", 0.05))

    epochs = settings.get("epochs", 10)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(test_features, dtype=torch.float32)
        test_y = torch.tensor(test_labels, dtype=torch.long)
        logits = model(test_x)
        eval_loss = float(criterion(logits, test_y).item())
        probs = torch.softmax(logits, dim=1)[:, 1].tolist()
        preds = logits.argmax(dim=1).tolist()
        correct = sum(p == t for p, t in zip(preds, test_labels, strict=True))
        accuracy = float(correct / len(test_labels))

    result = {
        "metrics": {
            "accuracy": accuracy,
            "loss": eval_loss,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "seed": seed,
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    predictions = {"test_ids": test_ids, "probabilities": probs}
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
