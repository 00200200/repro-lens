# Verification contract

For checking an edit against an earlier experiment, retain reports from both revisions
and use [before-and-after comparison](agent-review.md). Two runs of the current revision
alone do not establish agreement with the earlier output. The
[scikit-learn](../examples/sklearn_review/) and [XGBoost](../examples/xgboost_review/)
CPU examples run this comparison on actual local fits.

```toml
[tool.repro-lens.verify]
command = ["{python}", "train.py", "--output", "{output}"]
inputs = ["train.py", "config.toml", "data/*.csv", "uv.lock"]
metrics = ["accuracy"]
artifacts = ["predictions.json"]
result = "result.json"
timeout = 60
atol = 1e-12
rtol = 1e-9
```

Command is an argv array, never a shell string. `{output}` is a fresh absolute directory
for each run. `{python}` is the verifier's interpreter. For another environment, use
its reviewed interpreter or `uv run --locked --no-sync python ...` as the template does.
Call `uv sync` first. The runner does not provision environments or constrain network.

The result JSON contains `{"metrics": {"accuracy": 0.9}}` and optionally `runtime`
with relevant package versions. Declared metric keys must be finite numbers; boolean,
string, NaN and Infinity values fail. At least one metric or artifact is mandatory.
Runtime metadata must agree as typed JSON across both runs; for example, `true` and
`1` are different metadata values even though Python considers them equal.

Object keys must be unique within each JSON object. Nonfinite numbers are rejected
throughout the result, including runtime metadata and undeclared metrics. This includes
exponents such as `1e999` that overflow Python's float range and values such as `1e-400`
that underflow to zero. These validation errors stop verification and preserve an error
report, the original result JSON and run logs. The result file must not exceed 2,000,000
bytes.

Numeric comparisons use `abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))`, with
exact arithmetic on the parsed values. Integer metrics keep their full precision;
zero tolerances require numeric equality, including for counts above `2**53`.
Decimal literals in JSON and TOML still undergo Python's usual binary floating-point
rounding during parsing; comparison introduces no further rounding. Literals that
underflow to zero are rejected rather than stored as `0.0`. Artifacts use
exact SHA-256. Undeclared metrics are not compared. Each input glob must match at
least one file.

Inputs/artifacts must remain inside their project/output roots; escaping symlinks are
rejected. Inputs are hashed before and after each run. The runner does not prove that
all declared inputs are consumed, track external state, or detect transient modifications
that are restored before hashing.

Reports retain commands, successful exit statuses, timings, metrics, hashes, logs, Git
commit/dirty state, runner platform and optional child runtime metadata. Failed runs
retain logs and an error. Child runtime metadata is self-reported, not attested.

Exit 0: matched; exit 1: mismatch; exit 2: configuration/execution error. Fresh output
directories prevent stale metrics from hiding failure. POSIX timeouts kill the process
group; on Windows only the direct process is killed. The runner is not a sandbox,
does not limit GPU use or remove inherited credentials.
