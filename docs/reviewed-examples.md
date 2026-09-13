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

## Framework expansion: four public Python examples

On 2026-09-13, the implementation introducing R104–R110 was also checked against
four hand-picked public files. All four were read in full and scanned without
installing their frameworks or running them. Exact commits, SHA-256 hashes,
download URLs, findings and interpretations are retained in the
[reviewed-source manifest](../examples/framework_checks/reviewed_sources.json).

| Reviewed source | Findings | Interpretation |
| --- | --- | --- |
| [XGBoost GLM](https://github.com/dmlc/xgboost/blob/2ddf6aefe6cd334e7639a68d1b811503c5fbe084/demo/guide-python/generalized_linear_model.py) | R104 warning, line 42 | The dictionary is assigned once and used once in a later direct call. The scanner resolves `gblinear` without an updater, selecting the documented nondeterministic `shotgun` algorithm. |
| [LightGBM sklearn interface](https://github.com/microsoft/LightGBM/blob/d02a01ac6f51d36c9e62388243bcb75c3b1b1774/examples/python-guide/sklearn_example.py) | Two R105 reviews, lines 24 and 71 | Both constructors omit deterministic mode. No output instability was established; default seeds are not treated as uncontrolled entropy. |
| [PyTorch single-GPU tutorial](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/distributed/ddp-tutorial-series/single_gpu.py) | R106, line 58 | Explicitly shuffled loading omits a generator. The custom dataset and global RNG need context. |
| [Lightning autoencoder](https://github.com/Lightning-AI/pytorch-lightning/blob/655a3a91828694b9bfc241177f474741e483aeaa/examples/pytorch/basics/autoencoder.py) | None | The split has a seeded generator and loaders do not explicitly shuffle. `LightningCLI` constructs the Trainer indirectly, outside R109 coverage. |

After adding single-use named dictionary resolution: three nonblocking review items
and one warning (the initial framework implementation reported R190 for the XGBoost
call). This small convenience sample does not measure recall or a general
false-positive rate. The warning identifies a documented algorithm choice, not an
observed mismatch, and Lightning's clean result does not certify its Trainer policy.
TensorFlow's new rule was checked against its primary API
contract and synthetic cases; no external TensorFlow project was evaluated here.
No upstream defect, framework execution result or independent adoption is claimed.

To repeat, fetch the exact URLs in the manifest, verify each SHA-256, and scan the
downloaded `.py` files with this development checkout. Record `git rev-parse HEAD`
alongside the report: the v0.2.0 release predates these checks. The seven-pair
[local demo](../examples/framework_checks/) provides an offline behavioral check,
including positive and negative examples for every new rule.
