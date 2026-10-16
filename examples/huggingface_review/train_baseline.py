"""Fine-tune a classification model using Hugging Face Transformers.

This baseline script contains common reproducibility pitfalls:
1. Dataset shuffling omits an explicit seed (R127).
2. TrainingArguments omits data_seed and full_determinism=True (R117).
"""

from datasets import Dataset
from transformers import TrainingArguments

# Pitfall 1: dataset shuffling without an explicit seed
raw_data = {"text": ["sample A", "sample B", "sample C", "sample D"], "label": [0, 1, 0, 1]}
dataset = Dataset.from_dict(raw_data)
shuffled_dataset = dataset.shuffle()

# Pitfall 2: TrainingArguments omitting data_seed and full_determinism=True
training_args = TrainingArguments(
    output_dir="./results",
    seed=42,
    num_train_epochs=1,
)
