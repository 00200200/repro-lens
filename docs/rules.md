# Rules and limits

| Code | Level | Meaning |
| --- | --- | --- |
| R101 | warning | A supported randomized sklearn call lacks explicit non-None random_state |
| R102 | warning | A new NumPy RNG lacks an explicit seed/RNG argument |
| R103 | warning | A new Python Random instance lacks an explicit seed |
| R104 | warning | XGBoost gblinear selects the nondeterministic shotgun updater |
| R105 | review | LightGBM determinism, device or histogram settings need review |
| R106 | review | PyTorch data sampling has no explicit seeded generator; global control is unresolved |
| R107 | warning | PyTorch cuDNN benchmarking is explicitly enabled |
| R108 | warning | TensorFlow explicitly initializes a nondeterministic RNG |
| R109 | review | Lightning Trainer does not request strict determinism without benchmarking |
| R110 | review | PyTorch deterministic algorithms are disabled or warning-only |
| R111 | review | NumPy's global RNG is used in a file that never seeds it |
| R112 | review | Python's global random module is used in a file that never seeds it |
| R113 | review | PyTorch's global RNG is used in a file that never seeds it |
| R114 | review | TensorFlow's global RNG is used in a file that never seeds it |
| R115 | review | pandas `sample()` has no random_state in a file that never seeds NumPy |
| R190 | review | Dynamic arguments or framework configuration prevent a decision |
| P201 | error | A file required by this project's policy is missing |
| P202 | error | Project configuration is invalid |
| P203 | error | Verification inputs/configuration cannot be resolved safely |
| S901 | error | Invalid suppression, unknown rule or missing justification |
| S902 | error | A Python file failed syntax or scope validation |

`--fail-on warning` is the default. Review items never fail the hook; choose
`--fail-on error` when warnings should remain advisory.

`--format sarif` writes SARIF 2.1.0 with one rule entry per code above; errors, warnings
and review items become SARIF `error`, `warning` and `note` results. `--format github`
prints GitHub Actions `::error`, `::warning` and `::notice` commands. In both formats a
notebook finding points at the notebook file, because a cell line has no line in the
`.ipynb` JSON; the cell and line are the start of the message.

See [framework coverage](frameworks.md) for R104–R115: exact APIs, primary sources,
passing examples and limits. These rules share the CLI, hook and skill engine.

Supported sklearn APIs are the explicit registry in `analysis.py`:

| Area | APIs |
| --- | --- |
| Splits and search | train_test_split, KFold, StratifiedKFold, StratifiedGroupKFold, ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, learning_curve, RandomizedSearchCV |
| Linear models | SGDClassifier/Regressor, SGDOneClassSVM, PassiveAggressiveClassifier/Regressor, RANSACRegressor |
| Trees and ensembles | DecisionTree, ExtraTree, RandomForest, ExtraTrees, GradientBoosting and Bagging Classifier/Regressor; IsolationForest, RandomTreesEmbedding |
| Neural networks | MLPClassifier/Regressor, BernoulliRBM |
| Clustering and mixtures | KMeans, MiniBatchKMeans, BisectingKMeans, GaussianMixture, BayesianGaussianMixture |
| Transformers | LatentDirichletAllocation, TSNE, RBFSampler, Nystroem, GaussianRandomProjection, SparseRandomProjection |
| Utilities and datasets | permutation_importance, utils.shuffle, utils.resample, make_classification, make_regression, make_blobs |

KFold, StratifiedKFold, StratifiedGroupKFold and learning_curve are checked with
shuffle=True. train_test_split and SGDOneClassSVM shuffle by default and are not
flagged with shuffle=False. SGDClassifier/Regressor and PassiveAggressiveClassifier/
Regressor also use random_state for the validation split when early_stopping=True
([`BaseSGD._make_validation_split`](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/linear_model/_stochastic_gradient.py#L263-L290)),
so they are flagged when shuffle or early_stopping is enabled. With shuffle=False and
an unresolved early_stopping (or shuffle), they are R190 review items unless an explicit
random_state is passed. KMeans with a non-string
`init` (a centroid array, callable or variable) is an R190 review item, since fixed
centroids do not use random_state. Every other API is flagged whenever random_state
is missing, following its scikit-learn 1.9 documentation.

APIs seeded by default, such as Perceptron and permutation_test_score (both
`random_state=0`), are not flagged. Neither are APIs that use random_state only for
some arguments: PCA and TruncatedSVD solvers, FastICA without `w_init`,
HistGradientBoosting binning or early stopping, SpectralClustering, and AdaBoost with
a custom estimator. This registry does not model every library version or every
estimator's parameter conditions.

R102 covers `numpy.random.default_rng`, `RandomState` (including
`numpy.random.mtrand.RandomState`), `SeedSequence`, and the public BitGenerator
constructors `PCG64`, `PCG64DXSM`, `MT19937`, `Philox` and `SFC64`. NumPy's
[Generator guide](https://numpy.org/doc/stable/reference/random/generator.html)
constructs `Generator(PCG64())`; with `seed=None` those constructors draw OS
entropy, the same as `default_rng()`. The finding is on the BitGenerator call.
`SeedSequence()` with omitted/`None` `entropy` likewise draws OS entropy
([SeedSequence](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html));
`spawn_key=` alone does not pin the pool. `Philox(key=...)` counts as entropy
control; `counter=` alone does not (see
[Philox](https://numpy.org/doc/stable/reference/random/bit_generators/philox.html)).
`numpy.random.Generator(...)` itself is not flagged: it requires a BitGenerator.

RNG findings are screening warnings: omitted random_state can be deliberate with
controlled upstream global RNG state. The checker does not infer that state. Pass an
explicit seed/RNG or justify the actual policy on the call's first line:

```python
split(X)  # repro-lens: ignore[R101] -- Shared RNG is seeded by the reviewed entrypoint.
```

Suppressed findings remain listed in JSON. No seed autofixes are made. Explicit seed
expressions are accepted, but their runtime values are not evaluated.

Imports/aliases and common shadowing are tracked conservatively. Function-local names
come from [Python's symbol tables](https://docs.python.org/3/library/symtable.html):
parameters or assignments in a nested function, class, lambda or comprehension do not
hide an import used by the enclosing function. Class attributes also do not hide an
import used by a method, lambda, or comprehension expression in that class; those
names resolve in the enclosing function or module. Method defaults and a
comprehension's first iterator still use the class body. Actual function locals
shadow outer imports even before assignment. Invalid scope declarations are
reported as S902.

Lambda default arguments are checked in the enclosing scope, following
[Python's default argument semantics](https://docs.python.org/3/tutorial/controlflow.html#default-argument-values).
For example, `lambda rng=random.Random(): rng` reports R103 when `random` is the
imported module. The lambda's parameters shadow imports in its body; parameters of
a nested lambda in a default do not hide imports used by the outer lambda.

Calls in function/class decorators, class bases and
class keyword arguments are also checked in the enclosing scope. For example,
`@configure(rng=np.random.default_rng())` reports R102 when `np` is the imported
NumPy module. Definition annotations and implicit calls made by decorators or
metaclasses are not analyzed.

Jupyter notebooks (`.ipynb`, nbformat 4) are also
checked. Code cells are joined in file order, so an import in one cell resolves in
later cells. Findings report the cell number (counting all cells) and the line within
that cell, for example `explore.ipynb:cell 3:2:15`. Line magics, shell commands (`!`),
`x = !cmd` captures and `obj?` help lines are skipped. As in IPython 7.34, 8.12 and
9.17, such a command ends at its line unless the line ends with a backslash; brackets
and quotes inside it do not continue it, so the next line is checked as Python. Cell
magics other than `%%time`, `%%timeit`, `%%capture` and `%%prun` are skipped; the bodies
of those four are checked. A cell that is not valid Python, including a magic whose
arguments continue over brackets on the next line, is reported as S902 with its
location, and the other cells are still checked. Execution order,
execution counts and saved outputs are not analyzed, and `.ipynb_checkpoints` is
skipped.

Wrappers, monkeypatches, star/dynamic imports, complex control flow and imports after
function definitions can be missed. This is not a whole-program analyzer. Only `.py`
and `.ipynb` files are inspected; escaping paths and generated directories are skipped. Git file discovery honors
exclusions; without Git, default directory exclusions and configured globs apply.

Project policy is optional; no folder layout is universally required:

```toml
[tool.repro-lens]
required-files = ["uv.lock", "docs/reproducing.md", "data/README.md"]
exclude = ["vendor/**", "examples/intentionally_broken/**"]
```

The checker verifies existence, not prose accuracy or lockfile consistency. Use
`uv lock --check` for the latter. Selected files limit code scanning; CI should also
run a full scan. Folder existence does not establish reproducibility.
