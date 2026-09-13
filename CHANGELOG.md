# Changelog

## 0.1.1 — 2026-09-13

- Preserve integer metric precision and avoid overflow during tolerance comparisons ([#2](https://github.com/00200200/repro-lens/pull/2)).
- Keep function-local bindings separate from nested scopes, reducing missed findings and false alarms ([#3](https://github.com/00200200/repro-lens/pull/3)).
- Reject duplicate JSON keys and nonfinite values throughout experiment results while retaining error reports and logs ([#4](https://github.com/00200200/repro-lens/pull/4)).
- Add a walkthrough, installation from a versioned Git tag and an isolated pre-commit configuration.

## 0.1.0 — 2026-09-13

Initial release: Python randomness checks, explicit project policy, local experiment
replay, a pre-commit hook, an Agent Skill and a runnable scikit-learn template.
