# Replay and compare a real scikit-learn fit

From the repository root, with Git, uv and Python 3.11–3.13:

```bash
uv run python examples/sklearn_review/demo.py
```

Unlike the static framework examples, this command installs pinned scikit-learn and
NumPy dependencies in separate local environments and actually trains six small
decision trees. It uses 16 committed synthetic XOR-like rows. No GPU, remote dataset
or API key is needed. The first run needs package downloads.

The example makes fresh copies under `.repro-lens/sklearn-demo/` and leaves the
versioned fixture unchanged. Each copy has its own locked environment. The normal
Repro Lens engine runs each variant twice and compares its retained reports:

| Variant | Two runs | Compared with baseline |
| --- | --- | --- |
| Baseline, tree depth 3 | matched | — |
| Equivalent feature selection | matched | matched |
| Tree depth reduced to 1 | matched | mismatch |

Each fit writes predictions, training accuracy and runtime versions. The refactor must
change only `train.py`; the shallower model must change only `config.toml`. The demo
exits nonzero if replay, comparison status or the expected changed inputs disagree.
On the initial macOS run, training accuracy was 0.9375 for the first two variants
and 0.625 for the depth-1 stump. Exact values are not a cross-platform promise.

This uses actual scikit-learn training, with scripted edits on a synthetic fixture.
Training accuracy illustrates output changes; it is not a validation score, a model
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

The code uses scikit-learn 1.6.1's `DecisionTreeClassifier` with an explicit
`random_state` and `max_depth`. A depth-1 stump cannot separate the XOR-like labels;
deeper trees can. References: [1.6 tree API](https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
and [decision tree reproducibility](https://scikit-learn.org/1.6/modules/tree.html#randomness-control).
See the [verification contract](../../docs/verification.md) for the report's limits.
