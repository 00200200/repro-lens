# Framework coverage

Repro Lens v0.3.1 includes R104–R114. Install the versioned release using the
[installation guide](../README.md#install), then run `repro-lens check --root YOUR_PROJECT`.
The scanner needs no ML dependencies and never executes the scanned file.
Named-dictionary resolution described below is included in v0.3.1.

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
from torch.utils.data import DataLoader, RandomSampler, random_split

random_split(dataset, lengths)  # R106 review: global RNG may be controlled elsewhere.
random_split(dataset, lengths, generator=torch.Generator().manual_seed(experiment_seed))
DataLoader(dataset, shuffle=True, generator=torch.Generator())  # R106: constructor is unseeded.
DataLoader(dataset, shuffle=True, generator=seeded_generator)
DataLoader(dataset, sampler=RandomSampler(dataset))  # R106 at the sampler, not the loader.
RandomSampler(dataset, generator=torch.Generator().manual_seed(experiment_seed))
torch.backends.cudnn.benchmark = True  # R107 warning: algorithm-selection risk.
torch.use_deterministic_algorithms(True, warn_only=True)  # R110 review.
```

R106 checks `random_split`, explicitly shuffled `DataLoader` calls, and the stdlib
`RandomSampler`, `WeightedRandomSampler` and `SubsetRandomSampler` constructors for a
seeded generator, including positional arguments. The defining `dataset`, `dataloader`
and `sampler` submodule imports are recognized too. `torch.Generator()` draws OS entropy
until `manual_seed` is given a non-None argument, so `generator=torch.Generator()` and
`generator=torch.Generator().manual_seed(None)` are review items — the same as
omitting the generator. See the [Generator API](https://docs.pytorch.org/docs/2.8/generated/torch.Generator.html)
and [Sampler signatures](https://docs.pytorch.org/docs/2.8/data.html) (2.8 reference).
A variable such as `generator=rng` is still accepted; its seed is
not proven. `manual_seed(*args)` stays silent, since the expansion may carry a seed.
`SequentialSampler` is not random. Global `torch.manual_seed`, transforms, user-defined
`Sampler` subclasses and worker callbacks are not followed. `DataLoader(..., sampler=x)`
does not inspect `x`; the finding is on the sampler constructor.

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

## Global RNG state — R111–R114 (review)

```python
import numpy as np
import torch

np.random.shuffle(indices)  # R111 review: nothing in this file seeds NumPy's global RNG.
torch.randn(64, 32)  # R113 review: nothing in this file calls torch.manual_seed.

np.random.seed(experiment_seed)
torch.manual_seed(experiment_seed)
np.random.shuffle(indices)  # No finding.
torch.randn(64, 32)  # No finding.
```

These rules look at one file as a whole. A call that draws from a library's global
RNG is a review item when the same file contains no call that seeds that library:

| Library | Uses (examples) | Seeding calls recognized |
| --- | --- | --- |
| NumPy (R111) | `numpy.random.rand`, `randn`, `randint`, `choice`, `shuffle`, `permutation`, `normal`, `uniform`, … (legacy global functions) | `numpy.random.seed`, `numpy.random.set_state` |
| Python (R112) | `random.random`, `randint`, `choice`, `choices`, `sample`, `shuffle`, `uniform`, `gauss`, … | `random.seed`, `random.setstate` |
| PyTorch (R113) | `torch.rand`, `randn`, `randint`, `randperm`, `*_like`, `bernoulli`, `multinomial`, `normal`, `poisson`, `torch.nn.init.*_` | `torch.manual_seed`, `torch.random.manual_seed`, `torch.set_rng_state`, `torch.random.set_rng_state` |
| TensorFlow (R114) | `tf.random.normal`, `uniform`, `truncated_normal`, `shuffle`, `categorical`, `gamma`, `poisson` | `tf.random.set_seed`, `tf.compat.v1.set_random_seed` |

Helpers that seed several libraries count for each of them: `seed_everything` from
`lightning`, `lightning.pytorch`, `lightning.fabric` and `pytorch_lightning`
(Python, NumPy, PyTorch), `transformers.set_seed` and `accelerate.utils.set_seed`
(same three), and `keras.utils.set_random_seed` / `tf.keras.utils.set_random_seed`
(Python, NumPy, TensorFlow). `torch.cuda.manual_seed_all` alone does not seed the
CPU generator and is not counted.

A seeding call counts only when it has an explicit non-None argument
(`np.random.seed(seed)`, `random.seed(seed)`, `torch.manual_seed(seed)`,
`seed_everything(seed)`). `seed()` and `seed(None)` draw OS or clock entropy — the
same as omitting the call — so a later global draw in that file is still a review
item. [`numpy.random.seed`](https://numpy.org/doc/stable/reference/random/legacy.html)
and [`random.seed`](https://docs.python.org/3/library/random.html#random.seed)
document that omitted/`None` seeds are not a fixed experiment seed; Lightning's
`seed_everything(None)` can also read `PL_GLOBAL_SEED` or generate a random seed.
An expression such as `seed(config.seed)` is accepted without evaluating it. `**`
expansions of a seeder stay silent, since they may carry that argument.

A PyTorch call with an explicit non-None `generator=` and a TensorFlow op with an
explicit non-None `seed=` do not use the global state and are not reported: TensorFlow
documents that an operation seed alone yields a repeatable sequence. Calls with `**`
expansions stay silent, since the expansion may carry that argument. Generator objects
(`np.random.default_rng(seed).shuffle`, `random.Random(seed).choice`) are method
calls on a variable and are outside these rules; R102/R103 cover construction of
`default_rng`/`RandomState`/BitGenerators and `random.Random`.

Seeding is recognized anywhere in the file, even after the use or inside another
function, and never across files: a project that seeds in `main.py` and draws in
`data.py` gets a review item in `data.py`. Order of execution, seeding through a
framework's CLI or environment, and per-process seeding of DataLoader workers are not
analyzed. Primary references: [NumPy legacy random](https://numpy.org/doc/stable/reference/random/legacy.html),
[`random.seed`](https://docs.python.org/3/library/random.html#random.seed),
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html) and
[`tf.random.set_seed`](https://www.tensorflow.org/api_docs/python/tf/random/set_seed).

## pandas sampling — R115 (review, development checkout)

```python
import pandas as pd

train = frame.sample(frac=0.8)  # R115 review: no random_state, NumPy never seeded here.
train = frame.sample(frac=0.8, random_state=seed)  # No finding.
```

With `random_state=None`, pandas' `DataFrame.sample`, `Series.sample` and
`GroupBy.sample` draw from `numpy.random`, the legacy global state
([pandas 3.0 `random_state` helper](https://github.com/pandas-dev/pandas/blob/v3.0.0/pandas/core/common.py)).
So a file that seeds NumPy (`np.random.seed`, `np.random.set_state` or a cross-library
helper listed above) gets no finding, and an explicit non-None `random_state` is
accepted without evaluating it.

The receiver's type is not resolved. A call is treated as pandas sampling only when
the file imports pandas, the method is `.sample`, and the arguments fit pandas'
signature: only keywords, among `n`, `frac`, `replace`, `weights`, `axis`,
`ignore_index` and `random_state`. This keeps `random.sample(items, 3)`,
`rng.sample(population, k=3)`, `dist.sample((5,))`, scikit-learn's `gmm.sample(100)` and
`kde.sample(n_samples=44)`, and Polars' `seed=` out of the rule. It also means
positional calls such as `frame.sample(5)`, `frame.sample()` and `**` expansions are
not reported. A file that handles
DataFrames without importing pandas is not checked.

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
and justified [suppressions](rules.md). Only `.py` and `.ipynb` files are discovered. Framework
versions are not detected, and these references are not a compatibility certification
across releases. Passing snippets and source reviews are maintainer validation, not
an independent false-positive benchmark or a SOTA result.
