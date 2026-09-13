# A repeatable change can still change the result

This demo uses the same `verify` and `compare` engine as the CLI and reproducibility skill. It runs a tiny threshold classifier against eight committed synthetic samples. No ML libraries, model downloads or API key are needed.

From the repository root, with Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
uv run --no-dev python examples/agent_review/demo.py
```

Expected output, followed by the path to the retained evidence:

```text
Scenario             verify     compare with baseline
baseline             matched    -
refactor             matched    matched
changed threshold    matched    mismatch
changed tolerance    matched    not_comparable
```

Every variant is repeatable by itself. Only the refactor preserves the baseline's declared outputs under the same verification contract.

| Edit | Observed result |
| --- | --- |
| Replace the integer conversion with an equivalent conditional | Accuracy stays at 1.0 and the prediction file is identical; the code change appears in the input hashes |
| Raise the classification threshold from 0.5 to 0.7 | Accuracy falls from 1.0 to 0.75 and two predictions change; before/after comparison reports `mismatch` |
| Also raise the accuracy tolerance from 0.0 to 0.3 | The contract changed, so comparison reports `not_comparable` and requires review |

The edits are scripted to demonstrate the checks. No coding agent is invoked, and these results are not an agent benchmark. For an actual agent-driven change, use the [reproducibility workflow](../../docs/agent-review.md) on your own experiment.

## Inspect the evidence

Each invocation creates a new directory under `.repro-lens/agent-demo/`. The source fixture is left intact. The directory contains the baseline and three edited experiment copies, eight process runs with logs and outputs, three comparison reports, and `summary.json` linking the reports. An unexpected status, accuracy or changed-input list makes the demo fail, so CI checks its advertised behavior.

To choose the parent directory for the retained evidence:

```bash
uv run --no-dev python examples/agent_review/demo.py --output /path/to/demo-evidence
```

Use the comparison paths printed in `summary.json` to inspect the actual differences. Each comparison contains `before.path` and `after.path`; these are the two retained verification reports accepted by the CLI:

```bash
uv run --no-dev repro-lens compare /path/to/baseline/report.json /path/to/changed/report.json
```

The individual `compare` command exits 1 for `mismatch` or 2 for `not_comparable`. The demo exits 0 when all three expected outcomes occur. A match applies only to the declared outputs and recorded environment; it does not establish scientific validity.
