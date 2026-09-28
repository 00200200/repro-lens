## Description
<!-- Briefly describe what changes you are introducing and the problem they solve. -->

Closes #<!-- Link the issue this PR fixes, e.g. Closes #42 -->

## Type of Change
- [ ] 🚀 New feature / rule (e.g. adding detection for a framework RNG function)
- [ ] 🐛 Bug fix (fixes false positive, false negative, or edge case)
- [ ] ⚡ Performance improvement (faster AST traversal, reduced memory)
- [ ] 📚 Documentation update (docs/rules.md, examples)
- [ ] 🧪 Tests (adds unit or integration tests)

## Validation & Verification
<!-- Describe testing steps taken to verify the changes. -->
- [ ] `uv run pytest` passes (all tests green).
- [ ] `uv run ruff check .` reports no lint errors.
- [ ] `uv run ruff format --check .` confirms clean code formatting.
- [ ] If adding a new rule: rule documented in `docs/rules.md` with examples and severity.
