# Framework coverage

Repro Lens v0.3.0 adds R104–R110. Install the versioned release using the
[installation guide](../README.md#install), then run `repro-lens check --root YOUR_PROJECT`.
The scanner needs no ML dependencies and never executes the scanned file.
Named-dictionary resolution described below is a development addition after v0.3.0;
use `uv tool install .` from the current checkout to try it.

These are selected API checks, not complete framework support. `warning` means a
documented source-level risk, not an observed output mismatch. `review` is advisory
and never fails a commit. Run the [static examples](../examples/framework_checks/)
and use [verify/compare](agent-review.md) to collect evidence from an authorized experiment.

## XGBoost — R104

For actual CPU training and before/after output checks, run the
[XGBoost replay example](../examples/xgboost_review/). It uses the shared
`verify`/`compare` engine and a committed synthetic dataset.

```python
import xgboost as xgb

xgb.XGBRegressor(booster="gblinear", random_state=experiment_seed)  # R104
xgb.XGBRegressor(booster="gblinear", updater="coord_descent", random_state=experiment_seed)
```

`shotgun` is the default updater for `gblinear`; its parallel updates are documented
as nondeterministic. A seed does not resolve this choice. Review a different updater
before changing a scientific experiment. Tree boosters and an omitted seed are not
flagged by this rule: XGBoost already has a default seed.
An explicit `updater=None` also leaves the default in place, following the
[Python parameter setter](https://github.com/dmlc/xgboost/blob/v3.0.2/python-package/xgboost/core.py#L2199).

Supported: `XGBModel`, `XGBClassifier`, `XGBRegressor`, `XGBRanker`, `XGBRFClassifier`
and `XGBRFRegressor` in `xgboost`/`xgboost.sklearn`, plus `train`/`cv` in
`xgboost`/`xgboost.training`. `gblinear` is deprecated in XGBoost 3.3+; it remains
relevant to older experiments. See the [XGBoost parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-linear-booster-booster-gblinear)
(reviewed with documentation reporting 3.4.1).

## LightGBM — R105 (review)

```python
import lightgbm as lgb

lgb.LGBMClassifier(deterministic=True)  # Review histogram mode.
lgb.LGBMClassifier(deterministic=True, force_col_wise=True, random_state=experiment_seed)
```

Review an omitted/disabled `deterministic`, a non-CPU device with that flag enabled,
or enabled CPU determinism without exactly one of `force_col_wise`/`force_row_wise`.
Default seeds may already give repeatable results; omission is not a defect finding.
The primary `device_type` option takes precedence over `device`.

Supported: `LGBMModel`, `LGBMClassifier`, `LGBMRegressor`, `LGBMRanker` in
`lightgbm`/`lightgbm.sklearn` and `train`/`cv` in `lightgbm`/`lightgbm.engine`.
See [LightGBM's determinism guidance](https://lightgbm.readthedocs.io/en/stable/Parameters.html#deterministic)
(4.7.0 documentation). Different hardware/builds/versions still need separate validation.

## PyTorch — R106, R107, R110

```python
import torch
from torch.utils.data import DataLoader, random_split

random_split(dataset, lengths)  # R106 review: global RNG may be controlled elsewhere.
random_split(dataset, lengths, generator=torch.Generator().manual_seed(experiment_seed))
DataLoader(dataset, shuffle=True, generator=seeded_generator)
torch.backends.cudnn.benchmark = True  # R107 warning: algorithm-selection risk.
torch.use_deterministic_algorithms(True, warn_only=True)  # R110 review.
```

R106 checks `random_split` and explicitly shuffled `DataLoader` calls for a non-None
generator, including positional arguments. The defining `dataset` and `dataloader`
submodule imports are recognized too. Global `torch.manual_seed`, transforms, custom
samplers and worker callbacks are not followed. A supplied generator expression is
accepted; its seed/state is not proven.

R107 checks direct/annotated `cudnn.benchmark = True` assignments. Dynamic values and
later overrides are not traced. R110 reviews `use_deterministic_algorithms(False)` or
`warn_only=True` with mode enabled. See [PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
and [data-loading signatures](https://docs.pytorch.org/docs/2.8/data.html) (2.8 reference).
No CPU/GPU equivalence or worker determinism is established by these checks.

## TensorFlow — R108

```python
import tensorflow as tf

tf.random.Generator.from_non_deterministic_state()  # R108, even after a global seed.
tf.random.Generator.from_seed(experiment_seed)
```

This rule covers the explicit nondeterministic constructor, including its
`random.experimental`, `compat.v1.random` and `compat.v1.random.experimental` aliases.
It does not infer seeds for arbitrary `tf.random` operations, Keras layers or datasets,
or establish that op determinism is enabled. Restoring recorded generator state is
also valid. See [TensorFlow's Generator API](https://www.tensorflow.org/api_docs/python/tf/random/Generator#from_non_deterministic_state)
(2.16.1 reference) and [op determinism](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).

## Lightning — R109 (review)

```python
from lightning.pytorch import Trainer, seed_everything

Trainer()  # Review; external setup may already control the environment.
seed_everything(experiment_seed, workers=True)
Trainer(deterministic=True, benchmark=False)
```

Review omitted/disabled/`"warn"` determinism or enabled benchmarking. With
`deterministic=True`, omitted/None benchmarking is accepted because Trainer defaults
it to False. Recognized namespaces: `lightning`, `lightning.pytorch`,
`pytorch_lightning`, and their `trainer`/`trainer.trainer` imports (except
`lightning.trainer`). Fabric and `seed_everything` effects are not analyzed.
See the [Trainer 2.6.1 source contract](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.1/src/lightning/pytorch/trainer/trainer.py#L252).

## Dynamic arguments and limits

New framework rules resolve literal `**{...}` expansions and inline `train`/`cv`
parameter dictionaries, including nested dictionary overwrites. A named dictionary
is also resolved when it has one direct assignment (`params = {...}` or an annotated
assignment) and one use in a later direct call in the same module/function body:

```python
import xgboost as xgb

params = {"booster": "gblinear", "seed": experiment_seed}
model = xgb.train(params, data)  # R104 at this call, not at the assignment.
```

This also supports `Model(**options)` in the framework checks. Expressions inside
the dictionary remain unevaluated; an unknown relevant value still produces R190.
Following [Python's binding and mutation semantics](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements),
aliases, updates, repeated uses, rebindings, captures and global/nonlocal declarations
are not assumed safe. Even a later use/mutation or an unrelated nested binding with
the same name rejects resolution. Branch/loop/with/try bodies, comprehensions and
deferred consumers are unsupported. Star imports and recognized reflective namespace
access also disable this resolution. Indirect reflection and arbitrary external
mutation are outside the analyzer's model.

Other variable dictionaries, dynamic values and unknown expansions produce R190
when a relevant option cannot be resolved. Argument
validity is not type-checked; documented boolean options are recognized as Python
`True`/`False`, not coerced from strings or numbers. Existing sklearn rules keep their
more limited expansion handling.

All new rules share existing import/scope handling, exclusions, CLI/hook/skill outputs
and justified [suppressions](rules.md). Only `.py` files are discovered. Framework
versions are not detected, and these references are not a compatibility certification
across releases. Passing snippets and source reviews are maintainer validation, not
an independent false-positive benchmark or a SOTA result.
