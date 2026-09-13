# Check an agent's change against recorded experiment outputs

A refactor can produce a different result on every subsequent run while still being
perfectly repeatable. `verify` checks two runs of the current project. `compare`
checks whether their recorded outputs also agree with a retained baseline.

This workflow works through the CLI or the bundled reproducibility skill. An existing
coding agent handles code inspection, edits and interpretation. Repro Lens supplies
the checks and evidence; it does not start a model service. Report comparison is
available from v0.2.0 and accepts complete v0.1.x verification reports.

For a runnable example, try the [before/after demo](../examples/agent_review/). It exercises matching outputs, changed predictions and a changed tolerance using scripted edits and the same comparison engine. No ML libraries or API key are needed.

## Ask your coding agent

Load [the reproducibility skill](../skills/reproducibility/SKILL.md) in an agent that
supports [Agent Skills](https://agentskills.io/specification), then give it a scoped task:

> Use the reproducibility skill to refactor this training script while preserving the
> declared outputs. Local verification is authorized. Capture a baseline before editing,
> check the changed experiment and report the comparison with changed inputs.

The agent should preserve the experiment's existing RNG policy, data split and tolerances.
If a baseline cannot run, its review should identify the missing evidence. The skill
does not grant permission to use paid compute or publish a review.

## Run the underlying checks yourself

In a project with a reviewed [verification configuration](verification.md):

```bash
repro-lens verify --format json
```

Keep the returned `report_path` as the baseline. After the intended edit, run the same
command and keep the new report path. Each invocation retains separate evidence.
Then compare the two paths:

```bash
repro-lens compare /path/to/before/report.json /path/to/after/report.json
repro-lens compare /path/to/before/report.json /path/to/after/report.json --format json
```

Both reports must describe successful two-run matches. Comparison reads only those
two JSON files; it does not rerun training or read the artifact paths they contain.

## Interpret the result

| Status | Exit | Meaning |
| --- | --- | --- |
| `matched` | 0 | All recorded before/after run pairs agree under the unchanged contract and recorded environment |
| `mismatch` | 1 | At least one metric or artifact comparison differs |
| `not_comparable` | 2 | Verification configuration or recorded environment changed; output equivalence was not evaluated |
| `error` | 2 | A report is unreadable, invalid, incomplete or not a successful two-run match |

JSON includes hashes of the report files, their recorded Git states, changes to input
hashes, policy/environment changes and output differences. Text output lists changed
input names and the fields that require review; JSON retains their before/after values.

Input changes are expected during a refactor, so they are listed without automatically
failing comparison. Review them: matching outputs do not establish that changed data,
splits or dependencies are equivalent. The tool cannot classify an input as code or data.

Changing tolerances, declared metrics/artifacts, input patterns, command, timeout or
result filename produces `not_comparable`. So do differences in the recorded runner
environment or self-reported experiment runtime metadata. This strict contract avoids
using a relaxed tolerance or a dropped output to make a change appear to pass.
Metadata comparison preserves JSON value types: `true`, `1` and `1.0` are distinct.
Numeric metrics still use the documented tolerance formula.

## Comparison limits

Metrics use the same exact arithmetic and tolerance formula as `verify`; artifacts
use recorded SHA-256 hashes. All four before/after run pairs are compared. Tolerance
is not transitive, so checking only the first run on each side could miss a difference.

Each report is limited to 8,000,000 bytes. Duplicate keys, nonfinite JSON numbers,
missing output evidence and run values that contradict `matched` status are errors.
The tool does not authenticate reports, attest the environment, or rehash original
artifacts. It cannot detect unrecorded state or prove scientific validity.

For example, if both baseline runs score 0.90 and both changed runs score 0.85,
each `verify` can return `matched`. With unchanged zero tolerances, `compare` returns
`mismatch`. If the change also raises the tolerance, it returns `not_comparable`.
