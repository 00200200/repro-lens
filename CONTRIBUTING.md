# Contributing

For a new rule, provide a real failure mode, a primary-source reference, an explanation
of the API conditions in which it applies, and both failing and passing examples.
Benchmark false positives on reviewed code before enabling broad rules by default.

Use `uv sync --locked`, then `uv run pytest` and the Ruff checks documented in README.
Keep the checker stdlib-only. Framework-specific imports belong in experiments, not
in the scanner. Verify source and built-wheel installation when changing packaging.

Current useful contributions: notebook cell locations, stronger lexical scope
handling, versioned API signatures, dataset identity adapters and reviewed PyTorch
rules. These are not implemented or promised by the current release.
