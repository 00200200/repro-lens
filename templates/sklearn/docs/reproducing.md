# Reproducing the baseline

1. Run `uv sync --locked` from the project root.
2. Run `uv run pytest` to check the data and train/test partition contracts.
3. Run `repro-lens verify` with Repro Lens installed from its local source checkout.

The configured command executes training twice in separate processes. Each process
writes `result.json` and `predictions.json` into a fresh directory provided by the
runner. Numeric metrics use `atol=1e-12` and `rtol=1e-9`; predictions are compared
by exact SHA-256. Those settings are a documented contract, not a guarantee that
all machines produce identical floating-point results.

Inspect `.repro-lens/verify/<run-id>/report.json`, both run logs and their artifacts.
The report identifies declared code, configuration and data by hash, records Git
state, and includes Python, NumPy, SciPy and scikit-learn versions reported by the
experiment. Network downloads and GPU training are not part of this example.

`matched` supports local repeatability for these declared outputs only. Evaluation
quality, leakage, hyperparameter selection and robustness need separate scrutiny.
