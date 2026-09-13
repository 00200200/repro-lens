# Contributing

For a new rule, provide a real failure mode, a primary-source reference, an explanation
of the API conditions in which it applies, and both failing and passing examples.
Benchmark false positives on reviewed code before enabling broad rules by default.
The [reviewed examples](docs/reviewed-examples.md) show how to record source revisions
and distinguish an omitted argument from a demonstrated repeatability problem.

Use `uv sync --locked`, then `uv run pytest` and the Ruff checks documented in README.
Keep the checker stdlib-only. Framework-specific imports belong in experiments, not
in the scanner. Verify source and built-wheel installation when changing packaging.

Current useful contributions: notebook cell locations, stronger lexical scope
handling, version-aware API signatures, dataset identity adapters and reviewed
extensions to the [framework checks](docs/frameworks.md). These are follow-up areas,
not promises of complete coverage. New rules need both positive and negative evidence.

## Contribute an example

Start from the [example catalog](examples/README.md) and check open issues and PRs
for overlapping work. A small, useful case is enough; you do not need to add a rule.

For a runnable example, include:

- The question it answers and one command to run from the repository root.
- Required Python/framework versions, downloads and CPU/GPU requirements.
- Small synthetic or redistributable inputs, with their origin and license.
- Expected findings or comparison outcomes, including a case that should pass.
- Where evidence is retained and what the result does not establish.

Keep static examples as source text that the shared scanner reads. For replay,
use the existing verify/compare engine, isolate optional dependencies and keep
generated artifacts out of Git. Document intentional fixture files explicitly.
Make the example fail when its advertised outcome changes so CI can validate it.

For a bug report, code and actual versus expected output are enough to start;
a complete runnable demonstration can follow during review.
