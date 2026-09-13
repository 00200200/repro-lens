# Rules and limits

| Code | Level | Meaning |
| --- | --- | --- |
| R101 | warning | A supported randomized sklearn call lacks explicit non-None random_state |
| R102 | warning | A new NumPy RNG lacks an explicit seed/RNG argument |
| R103 | warning | A new Python Random instance lacks an explicit seed |
| R190 | review | Dynamic shuffle or expanded arguments prevent a decision |
| P201 | error | A file required by this project's policy is missing |
| P202 | error | Project configuration is invalid |
| P203 | error | Verification inputs/configuration cannot be resolved safely |
| S901 | error | Invalid suppression, unknown rule or missing justification |
| S902 | error | A Python file failed syntax or scope validation |

`--fail-on warning` is the default. Review items never fail the hook; choose
`--fail-on error` when warnings should remain advisory.

Supported sklearn APIs are the explicit registry in `analysis.py`: train_test_split,
KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit,
RepeatedKFold, RepeatedStratifiedKFold, RandomForestClassifier/Regressor,
ExtraTreesClassifier/Regressor, DecisionTreeClassifier/Regressor, make_classification,
make_regression and make_blobs. KFold/StratifiedKFold are checked with shuffle=True;
train_test_split with shuffle=False is not flagged. This registry does not model
every library version or every estimator's parameter conditions.

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
hide an import used by the enclosing function. Actual function locals shadow outer
imports even before assignment. Invalid scope declarations are reported as S902.

Wrappers, monkeypatches, star/dynamic imports, complex control flow and imports after
function definitions can be missed. This is not a whole-program analyzer. Notebook
cells and PyTorch are outside this version's coverage. Only `.py` files are inspected;
escaping paths and generated directories are skipped. Git file discovery honors
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
