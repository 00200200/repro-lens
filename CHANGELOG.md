# Changelog

## Unreleased

- Review pandas `sample()` calls without `random_state` in files that import pandas and never seed NumPy's global RNG (R115).
- Report unseeded NumPy BitGenerator constructors (`PCG64()`, `MT19937()`, …) as R102, the same as `default_rng()`.
- Review PyTorch `RandomSampler`, `WeightedRandomSampler` and `SubsetRandomSampler` constructors that omit a seeded generator (R106).
- Reject JSON numbers that underflow to zero (for example `1e-400`) in experiment results and retained reports, so they cannot match a true `0.0`.
- Add a scikit-learn CPU replay example that compares an equivalent feature-selection refactor with a shallower tree.
- Report static check findings as SARIF 2.1.0 (`--format sarif`) or GitHub Actions annotations (`--format github`), with rule links and notebook cell locations in the message.
- Add a composite GitHub Action that annotates pull requests, optionally writes SARIF for code scanning, and fails according to `fail-on`.
- State in the report limitations that global RNG seeding is recognized only within the same file.
- Review global RNG use when a seeder is called without an explicit non-None seed (`seed()`, `seed(None)`, `seed_everything()`).
- Treat `torch.Generator()` and `Generator().manual_seed(None)` as unseeded for shuffled DataLoader / random_split sampling (R106).
- Scan methods, lambdas and comprehension expressions using the enclosing function or module, not class-body attributes that share the same name.

## 0.3.1 — 2026-09-14

- Review global NumPy, Python, PyTorch and TensorFlow RNG use in files that never seed the corresponding library (R111–R114), including documented cross-library helpers.
- Check notebook code cells with cell and line locations, and keep IPython magics and shell commands outside the Python scan.
- Check calls in function/class decorators, class bases and class keyword arguments using their enclosing import bindings.
- Resolve directly assigned, single-use framework parameter dictionaries while retaining review findings for dynamic configuration.

## 0.3.0 — 2026-09-13

- Add seven documented framework checks for XGBoost, LightGBM, PyTorch, TensorFlow and Lightning, with advisory review findings for settings that may be controlled elsewhere.
- Resolve inline framework parameter dictionaries and literal keyword expansions while preserving uncertainty about dynamic configuration; include a dependency-free before/after scan demo.
- Check RNG calls in lambda default arguments and preserve outer imports when a default contains another lambda.

## 0.2.0 — 2026-09-13

- Compare retained verification reports before and after a code change, including metric and artifact differences, input changes and changes to the verification contract or recorded environment.
- Reject incomplete reports and run evidence that contradicts a claimed match; share strict JSON parsing and output comparison with the replay engine.
- Extend the reproducibility skill with baseline capture and change review, backed by the CLI comparison tool.
- Add an advisory rollout guide, a problem report form and reviewed examples of intentional global RNG control.

## 0.1.1 — 2026-09-13

- Preserve integer metric precision and avoid overflow during tolerance comparisons ([#2](https://github.com/00200200/repro-lens/pull/2)).
- Keep function-local bindings separate from nested scopes, reducing missed findings and false alarms ([#3](https://github.com/00200200/repro-lens/pull/3)).
- Reject duplicate JSON keys and nonfinite values throughout experiment results while retaining error reports and logs ([#4](https://github.com/00200200/repro-lens/pull/4)).
- Add a walkthrough, installation from a versioned Git tag and an isolated pre-commit configuration.

## 0.1.0 — 2026-09-13

Initial release: Python randomness checks, explicit project policy, local experiment
replay, a pre-commit hook, an Agent Skill and a runnable scikit-learn template.
