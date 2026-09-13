# Check one script, then replay an experiment

Install Python 3.11+, Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv tool install 'git+https://github.com/00200200/repro-lens.git@v0.2.0'
```

## See a finding without installing ML libraries

Create a new directory with a file named `split.py`:

```python
from sklearn.model_selection import train_test_split


def split_data(features, labels):
    return train_test_split(features, labels)
```

Run the checker from that directory:

```bash
repro-lens check
```

The diagnostic points to the call:

```text
split.py:5:12 R101 [warning] sklearn.model_selection.train_test_split has no explicit non-None random_state.
```

Exit code 1 tells pre-commit or CI that a warning needs attention. The scan does not
import scikit-learn, read training data or execute the function.

Make the experiment's seed an explicit input:

```python
from sklearn.model_selection import train_test_split


def split_data(features, labels, seed):
    return train_test_split(features, labels, random_state=seed)
```

Run `repro-lens check` again. The warning is gone. The caller still has to supply a
valid seed or RNG; the checker does not evaluate that value. If upstream RNG control
is intentional, document it using a [justified suppression](rules.md) instead.

## Run an actual repeatability check

From a directory where `repro-demo` does not already exist:

```bash
repro-lens init repro-demo --name repro_demo
cd repro-demo
uv sync --locked
uv run pytest
repro-lens check
repro-lens verify
```

This installs the example's ML dependencies and runs a baseline on a committed
synthetic dataset. The configuration declares its seed, inputs and expected outputs.
The runner executes training twice in separate processes and compares the declared
metrics and prediction file. Successful output starts with:

```text
Repro Lens: matched
```

The final line points to `report.json` in the evidence directory. The report records
input hashes, commands, environment details, both runs and any differences. Each run
retains its result JSON, prediction file, stdout and stderr. Keep this directory local unless you
have reviewed its contents for sharing.

A match describes these outputs in this environment. It does not establish scientific
validity, prevent data leakage or guarantee the same result on another machine. See
the [verification contract](verification.md) to adapt replay to your own experiment.
