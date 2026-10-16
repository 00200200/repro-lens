"""Reproducible fine-tuning using Hugging Face Transformers.

This hardened script addresses all reproducibility findings:
1. Environment determinism variable configured (CUBLAS_WORKSPACE_CONFIG).
2. Global RNG state seeded via transformers.set_seed().
3. Dataset shuffle explicitly seeded with seed=42.
4. TrainingArguments configures data_seed=42 and full_determinism=True.
"""

import os

from datasets import Dataset
from transformers import TrainingArguments, set_seed

# Step 1: Configure cuBLAS deterministic workspace
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Step 2: Seed Python, NumPy, and PyTorch (CPU and CUDA)
set_seed(42)

# Step 3: Explicitly seed dataset shuffle for deterministic sample ordering
raw_data = {"text": ["sample A", "sample B", "sample C", "sample D"], "label": [0, 1, 0, 1]}
dataset = Dataset.from_dict(raw_data)
shuffled_dataset = dataset.shuffle(seed=42)

# Step 4: Lock PyTorch backend and DataLoader shuffling
training_args = TrainingArguments(
    output_dir="./results",
    seed=42,
    data_seed=42,
    full_determinism=True,
    num_train_epochs=1,
)
