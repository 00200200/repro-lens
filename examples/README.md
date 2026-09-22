# Find an example

Choose the question you want to answer. These examples use the same engine as the
CLI and coding-agent skill; each links to its command, expected output and limits.

| I want to… | Start here | What runs | Expected result |
| --- | --- | --- | --- |
| Explore before/after code without installing | [Browser gallery](https://00200200.github.io/repro-lens/) | A static page with search and filters | Saved findings from the shared checker; no Python or training runs in the browser |
| Understand why two matching runs can still hide a changed result | [Before/after demo](agent_review/) | A tiny synthetic classifier; no ML dependencies or API key | Equivalent refactor matches; changed threshold mismatches; changed tolerance needs review |
| Check an actual scikit-learn refactor | [scikit-learn CPU replay](sklearn_review/) | Six small local trees, with pinned dependencies downloaded on first use | Equivalent feature selection matches; a depth-1 stump mismatches |
| Check an actual boosting refactor | [XGBoost CPU replay](xgboost_review/) | Six small local fits, with pinned dependencies downloaded on first use | Equivalent feature selection matches; changing tree depth mismatches |
| See which framework settings need attention | [Framework screening](framework_checks/) | Static analysis; no framework imports or training | Risky, unresolved and explicitly configured examples show different findings |
| Try the tool on my existing project | [Advisory rollout](../docs/trying-an-existing-project.md) | Static analysis of my source | Findings to review before enabling a blocking hook |
| Review a coding agent's edit to my experiment | [Agent workflow](../docs/agent-review.md) | My reviewed experiment command, before and after the edit | Reports show output changes and verification-contract changes |

## Run the smallest demo

With Python 3.11+, Git and [uv](https://docs.astral.sh/uv/getting-started/installation/),
start in a directory without an existing repro-lens checkout:

```bash
git clone https://github.com/00200200/repro-lens.git
cd repro-lens
uv run --no-dev python examples/agent_review/demo.py
```

The command prints the four outcomes and retains reports under
`.repro-lens/agent-demo/`. It runs local fixture code; it does not call an AI service.
See the [example's evidence guide](agent_review/#inspect-the-evidence) to inspect
what changed. These links follow the development branch; the latest published
release may not include every example or check.

## Share a useful case

A small example that exposes a missed finding or an unnecessary warning is useful
without a new feature. [Open a problem report](https://github.com/00200200/repro-lens/issues/new?template=problem.yml)
with the minimal code, tool/framework versions, actual output and expected result.
Do not include private data, credentials or confidential source.

To contribute a runnable example, follow the
[example contribution checklist](../CONTRIBUTING.md#contribute-an-example).
Our own fixtures demonstrate behavior; they do not establish independent adoption,
scientific validity or performance across all supported frameworks.
