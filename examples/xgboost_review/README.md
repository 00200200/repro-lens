# Replay and compare a real XGBoost fit

From the repository root, with Git, uv and Python 3.11–3.13:

```bash
uv run python examples/xgboost_review/demo.py
```

Unlike the static framework examples, this command installs pinned XGBoost and
NumPy dependencies in separate local environments and actually trains six small
models. It uses one CPU thread, 25 committed synthetic rows and 12 boosting rounds
per fit. No GPU, remote dataset or API key is needed. The first run needs package
downloads. Linux x86_64 uses the smaller `xgboost-cpu` package; other platforms use
the standard package, as described in the [XGBoost installation guide](https://xgboost.readthedocs.io/en/release_3.0.0/install.html#minimal-installation-cpu-only).

The example makes fresh copies under `.repro-lens/xgboost-demo/` and leaves the
versioned fixture unchanged. Each copy has its own locked environment. The normal
Repro Lens engine runs each variant twice and compares its retained reports:

| Variant | Two runs | Compared with baseline |
| --- | --- | --- |
| Baseline, tree depth 2 | matched | — |
| Equivalent feature selection | matched | matched |
| Tree depth reduced to 1 | matched | mismatch |

Each fit writes predictions, training RMSE and runtime versions. The refactor must
change only `train.py`; the shallower model must change only `config.toml`. The demo
exits nonzero if replay, comparison status or the expected changed inputs disagree.
On the initial macOS run, training RMSE was about 0.360819 for the first two variants
and 0.792179 for the shallower trees. Exact values are not a cross-platform promise.

This uses actual XGBoost training, with scripted edits on a synthetic fixture. The
training metric illustrates output changes; it is not a validation score, a model
quality benchmark or evidence of independent adoption. Zero tolerances and exact
prediction hashes apply within each recorded environment. The fixture does not
establish GPU, distributed, cross-version or whole-framework reproducibility.

## Inspect or adapt the experiment

The printed `summary.json` links to all three verification reports and the two
comparison reports. Each project copy retains commands, logs, predictions, metrics,
input hashes and package versions. Compare the paths directly if needed:

```bash
uv run repro-lens compare /absolute/baseline/report.json /absolute/changed/report.json
```

To adapt the example, copy the five files from `experiment/` to a new project.
Review the training code and `[tool.repro-lens.verify]`, run `uv sync --locked` in
that project, then `repro-lens verify --root /absolute/project` with the CLI installed.
Capture the baseline before making the intended edit and keep its report. Re-run
verification afterward and compare the report paths. Changing tolerances or the
output contract requires separate review.

The code uses XGBoost 3.0.2's `DMatrix`, `train` and `predict`, with explicit CPU,
histogram, thread and seed settings. References: [3.0 API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html)
and [training parameters](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html).
See the [verification contract](../../docs/verification.md) for the report's limits.
