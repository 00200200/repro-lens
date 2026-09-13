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
