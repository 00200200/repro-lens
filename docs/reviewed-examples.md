# Reading findings in context

This is a maintainer-run review of three hand-picked scikit-learn examples. It
illustrates explicit arguments and global RNG control; it is not an accuracy
benchmark, evidence of adoption, or an endorsement by scikit-learn. The examples
were read and scanned, not executed.

The scan used Repro Lens v0.1.1 at
`2031d2932d9c5c9e893092a9f5ec75057e000e85` and scikit-learn 1.7.2 at
`25dee604bae18205b01548348388baf7a1cdfe0e` on 2026-09-13.

| Source at the reviewed revision | Findings | Interpretation |
| --- | --- | --- |
| [OOB errors](https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/examples/ensemble/plot_ensemble_oob.py) | None | The dataset generator and all three forest constructors pass `RANDOM_STATE` explicitly. The scanner accepts that expression without evaluating it. |
| [Forest decision surfaces](https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/examples/ensemble/plot_forest_iris.py) | Four R101 warnings, lines 74–77 | The constructors omit `random_state`, but the example resets NumPy's global RNG before shuffling and fitting each model. Constructor warnings alone do not establish uncontrolled randomness. |
| [Bias–variance decomposition](https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/examples/ensemble/plot_bias_variance.py) | Two R101 warnings, lines 87–88 | The tree constructors omit `random_state`; the script initializes NumPy's global RNG earlier. The enclosing bagging estimator and subsequent RNG consumption also need context. |

All six warnings describe omitted arguments in code with upstream RNG control. No
repeatability failure was established. There is no basis here for reporting six
upstream bugs or changing the examples' statistical choices.

The scanner does not model global RNG state or all estimator behavior. Bagging and
AdaBoost are outside its current API registry even when a nested tree constructor
is recognized. A clean result on the OOB example does not establish whole-script
coverage or repeatability either.

## Repeat the static scan

With Repro Lens v0.1.1 installed, run in a fresh directory:

```bash
git clone --depth 1 --branch 1.7.2 https://github.com/scikit-learn/scikit-learn.git
git -C scikit-learn rev-parse HEAD
repro-lens check --root scikit-learn \
  examples/ensemble/plot_ensemble_oob.py \
  examples/ensemble/plot_forest_iris.py \
  examples/ensemble/plot_bias_variance.py \
  --format json --output sklearn-review.json --fail-on error
```

Confirm the checkout matches the revision above. Expected: three files inspected,
six R101 warnings and exit code 0 under the advisory threshold. No scikit-learn
installation or training run is needed. Later checker versions can produce different
results; retain their version alongside the report.

For a new project, use the [advisory rollout guide](trying-an-existing-project.md).
