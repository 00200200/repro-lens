# Try Repro Lens on an existing project

Start with one training script whose RNG policy you understand. You can check it
without installing its dependencies or running training. Install the CLI using the
[quickstart](quickstart.md) first.

## Start with an advisory scan

Replace the project directory and script name below:

```bash
repro-lens --version
repro-lens check --root /path/to/project train.py --fail-on error
repro-lens check --root /path/to/project train.py --format json --output audit.json --fail-on error
```

Paths after `check` are relative to `--root`. The report is written relative to your
current directory. `--fail-on error` keeps warnings advisory; configuration and
source errors still fail the command. Check `files_checked` in the JSON report:
a missing, excluded or unsupported file can leave you with zero files inspected.
This version checks `.py` files, not notebook cells.

## Review the findings

For each diagnostic, record the rule, location and relevant RNG policy:

| What you find | Next step |
| --- | --- |
| An unintended missing seed/RNG argument | Choose the experiment's RNG policy, make it explicit and rerun the scan |
| Intentional global RNG control | Check where the RNG is initialized and consumed; document a justified suppression if appropriate |
| A supported call missed or the wrong symbol flagged | Report a minimal example, including imports and surrounding scope |
| A `review` item with dynamic arguments | Inspect the actual arguments; the checker could not decide |
| No findings | Check coverage and proceed to execution evidence if repeatability matters |

Omitting `random_state` can be deliberate. scikit-learn's
[randomness guidance](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness)
explains why integers, RNG objects and `None` have different effects during repeated
fits and cross-validation. Inserting the same integer everywhere can change the
experiment. Repro Lens does not make that choice for you.

See [three reviewed scikit-learn examples](reviewed-examples.md) where an omitted
argument needs context before anyone should propose a fix. Suppression syntax and
coverage limits are in [rules.md](rules.md).

## Decide whether to enable the hook

After reviewing the first script, scan the whole project by omitting `train.py`.
Resolve or justify the findings before enabling the default warning threshold in
pre-commit. For an advisory rollout, the hook accepts the same option:

```yaml
repos:
  - repo: https://github.com/00200200/repro-lens
    rev: v0.3.0
    hooks:
      - id: repro-lens
        args: [--fail-on, error]
```

Remove `args` when warnings should block commits. This changes the failure threshold,
not the reported findings. Experiment replay remains a separate command.

## Add execution evidence when needed

The [verification guide](verification.md) describes how to declare a command, inputs,
metrics, artifacts and tolerances. Review that configuration before running
`repro-lens verify`: it executes the command with your permissions. Keep the report
from both successful and failed comparisons. A two-run match only describes the
declared outputs in that environment.

## Share useful feedback

[Report a problem](https://github.com/00200200/repro-lens/issues/new?template=problem.yml)
with the tool version, command, a small example and the behavior you expected.
For a randomness finding, include the imports and where the RNG is controlled.
For replay, include the relevant configuration and the first error or difference.

Review reports before sharing: they can contain absolute paths, commands and local
logs. A small code example is usually enough; training data and full experiment logs
are rarely needed to investigate a static finding.
