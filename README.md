# Repro Lens

Find uncontrolled randomness in Python ML code, then test whether an experiment produces the same outputs twice.

Repro Lens includes a CLI, a pre-commit hook, a reproducibility skill for coding agents, and a small scikit-learn project template. The checker needs only Python 3.11+; it never imports the project it scans.

## Install

```bash
git clone https://github.com/00200200/repro-lens.git
cd repro-lens
uv tool install .
```

## Check an existing project

```bash
repro-lens check --root /path/to/project
repro-lens check --root /path/to/project --format json
```

For example:

```python
from sklearn.model_selection import train_test_split

train_test_split(X, y)  # R101: no explicit random_state
```

The checker recognizes imported aliases, skips non-shuffled splits, and reports dynamic arguments as unresolved. Warnings can be justified with an inline comment. All rules and their limits are described in [docs/rules.md](docs/rules.md).

## Try a complete experiment

```bash
repro-lens init /tmp/iris-study --name iris_study
cd /tmp/iris-study
uv sync --locked
uv run pytest
repro-lens check
repro-lens verify
```

The example uses a committed synthetic CSV and a scikit-learn pipeline. `verify` launches two separate processes, compares metrics and prediction artifacts, and saves commands, logs, input hashes and results in `.repro-lens/verify/`.

Metrics use explicit tolerances. Artifacts use SHA-256. Missing outputs, nonfinite metrics, failed commands, timeouts and changed inputs are errors. See the [verification contract](docs/verification.md) to configure another experiment.

A clean static scan does not prove repeatability. A two-run match applies to the declared outputs in that environment. `verify` executes the configured command with your permissions; review it before running unfamiliar code.

## pre-commit

With the CLI installed, add this to an existing configuration:

```yaml
repos:
  - repo: local
    hooks:
      - id: repro-lens
        name: Repro Lens
        entry: repro-lens check
        language: system
        files: '\.(py|toml|lock)$'
        require_serial: true
```

The repository also supplies `.pre-commit-hooks.yaml` for isolated hook installation. The generated template includes Ruff, file checks, lockfile validation and this hook. Run the full scan and experiment verification in CI as well.

## Coding agents

[skills/reproducibility/SKILL.md](skills/reproducibility/SKILL.md) guides an existing coding agent through an audit or an authorized repair. Its launcher calls the same engine as the CLI. The complete repository is also a Codex plugin with `.codex-plugin/plugin.json`.

Example request:

> Use the reproducibility skill from this checkout to audit my ML project. Inspect the findings, fix reproducibility blockers and verify the local experiment.

The skill distinguishes source-level risks from observed execution results. It does not start a separate LLM service or require an API key. For standalone skill installation, install the CLI too; the full plugin checkout already includes its engine.

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Contributions need both a failure example and valid code that the rule must leave alone. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [architecture decision](docs/architecture.md).

Early release: selected scikit-learn APIs and Python/NumPy RNG construction are covered. Notebook cells, PyTorch, arbitrary wrappers and whole-program data flow are not yet supported. The package is not published to PyPI.

MIT licensed.
