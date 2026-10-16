# Hugging Face Fine-Tuning Reproducibility Review

This tutorial demonstrates how Repro Lens detects, reviews, and resolves reproducibility gaps in Hugging Face Transformers fine-tuning pipelines.

## The Problem: Hidden Non-Determinism in Hugging Face

Fine-tuning LLMs and Transformer models often appears repeatable when losses decrease steadily, yet subtle sources of variance cause run-to-run weight divergence:

1. **Unseeded Dataset Shuffling (`R127`)**:
   Calling `dataset.shuffle()` without an explicit `seed=...` uses arbitrary entropy. Not only does this change sample presentation order across runs, but it also invalidates Hugging Face Datasets' deterministic on-disk fingerprint cache.
2. **Missing `full_determinism` in `TrainingArguments` (`R117`)**:
   Passing only `seed=42` to `TrainingArguments` seeds Python, NumPy, and PyTorch, but **does not** enforce deterministic PyTorch operations or pin the internal DataLoader worker shuffling seed (`data_seed`). As a result, non-deterministic cuDNN convolution/GEMM algorithms and asynchronous gradient reductions run unconstrained.
3. **Missing CUDA Determinism Configuration**:
   PyTorch 2.0+ requires `CUBLAS_WORKSPACE_CONFIG=":4096:8"` or `":16:8"` in the environment before any CUDA context initializes; without it, certain atomic operations raise runtime errors or fall back to non-deterministic paths.

---

## Comparing Baseline vs Reproducible Code

### Baseline (`train_baseline.py`)

```python
from datasets import Dataset
from transformers import TrainingArguments

# Anti-pattern 1: dataset shuffle without seed
dataset = Dataset.from_dict(raw_data)
shuffled_dataset = dataset.shuffle()

# Anti-pattern 2: TrainingArguments without data_seed or full_determinism
training_args = TrainingArguments(
    output_dir="./results",
    seed=42,
    num_train_epochs=1,
)
```

Running `repro-lens check` on this baseline flags:
- `[R117 · review]`: `transformers.TrainingArguments omits full_determinism=True or an explicit non-None data_seed.`
- `[R127 · review]`: `shuffle() without an explicit seed causes non-deterministic ordering in Hugging Face Datasets.`

### Hardened Reproducible Implementation (`train_reproducible.py`)

```python
import os
from datasets import Dataset
from transformers import TrainingArguments, set_seed

# 1. Ensure deterministic cuBLAS memory allocation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 2. Seed Python, NumPy, and PyTorch (CPU and CUDA)
set_seed(42)

# 3. Explicitly seed dataset shuffle
dataset = Dataset.from_dict(raw_data)
shuffled_dataset = dataset.shuffle(seed=42)

# 4. Enforce strict PyTorch backend determinism and DataLoader seeds
training_args = TrainingArguments(
    output_dir="./results",
    seed=42,
    data_seed=42,
    full_determinism=True,
    num_train_epochs=1,
)
```

Running `repro-lens check` on `train_reproducible.py` passes with **0 findings**.

---

## Run the Screening Demo

To run the automated analysis comparing both versions:

```bash
python examples/huggingface_review/demo.py
```

Output:

```text
Analyzing Hugging Face baseline vs reproducible fine-tuning scripts...

Script                   Findings   Status
--------------------------------------------------
train_baseline.py        2          ⚠️ Detected R117, R127
   ↳ [R117 · review] line 17: transformers.TrainingArguments omits full_determinism=True or an explicit non-None data_seed.
   ↳ [R127 · review] line 14: shuffle() without an explicit seed causes non-deterministic ordering in Hugging Face Datasets.

train_reproducible.py    0          ✅ Fully Reproducible (0 findings)

Summary: Hugging Face fine-tuning script successfully reviewed and hardened.
```

---

## Verification Contract (`pyproject.toml`)

Repro Lens allows projects to declare verification policies under `[tool.repro-lens.verify]`:

```toml
[tool.repro-lens.verify]
command = ["python", "train_reproducible.py"]
inputs = ["train_reproducible.py", "pyproject.toml"]
metrics = []
artifacts = []
timeout = 60
atol = 1e-7
rtol = 1e-5
```

With `repro-lens` installed, run:

```bash
repro-lens check examples/huggingface_review/
```
