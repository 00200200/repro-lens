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
hash-inputs = true
```

Command is an argv array, never a shell string. `{output}` is a fresh absolute directory
for each run. `{python}` is the verifier's interpreter. For another environment, use
its reviewed interpreter or `uv run --locked --no-sync python ...` as the template does.
Call `uv sync --locked` first. Before the two runs, verify compares installed package
versions to `uv.lock` or `poetry.lock` when either file is present (P204) and stops if
they disagree. The runner does not provision environments or constrain network.

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
rejected. Inputs are hashed with SHA-256 (`hashlib` only) before and after each run.
When `hash-inputs = true`, a durable snapshot is written to
`.repro-lens/verify/inputs-sha256.json` after a successful two-run match or mismatch.
A later `verify` compares the live hashes against that snapshot and fails if a declared
input was added, removed or modified. The default `hash-inputs = false` still records
per-run SHA-256 maps in the report (for `compare`) without enforcing a durable snapshot.
The runner does not prove that all declared inputs are consumed, track external state,
or detect transient modifications that are restored before hashing.

Reports retain commands, successful exit statuses, timings, metrics, hashes, logs, Git
commit/dirty state, runner platform and optional child runtime metadata. Failed runs
retain logs and an error. Child runtime metadata is self-reported, not attested.
Verification also audits multi-threading determinism across BLAS and OpenMP environment
variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`). If unpinned or set to dynamic values,
it emits a warning in the verification report recommending explicit pinning (e.g.
`export OMP_NUM_THREADS=1` or a fixed thread count) to prevent floating-point reduction
order variations.

Exit 0: matched (or stable); exit 1: mismatch (or unstable); exit 2: configuration/execution error. Fresh output
directories prevent stale metrics from hiding failure. POSIX timeouts kill the process
group; on Windows only the direct process is killed. The runner is not a sandbox,
does not limit GPU use or remove inherited credentials.

## Multi-seed experiment variance & stability bounds (`--seeds`)

Running an experiment twice with the same seed verifies local determinism. Verifying
scientific stability across initialization seeds tests that metrics do not collapse or
vary beyond acceptable experimental boundaries:

```bash
repro-lens verify --seeds 42,43,44,45
```

When `--seeds` is provided:
1. The experiment command executes across each declared seed. `{seed}` in `command` is replaced by the current seed, and the environment variable `SEED` is exported.
2. Repro Lens calculates summary statistics for each declared metric: mean, sample variance ($s^2$ with Bessel's correction $N-1$), standard deviation ($s$), standard error of the mean ($\text{sem} = s / \sqrt{N}$), minimum, and maximum.
3. If `[tool.repro-lens.verify.stability]` is configured in `pyproject.toml`, metrics are checked against configured bounds (`max_std`, `max_variance`, `min_mean`, `max_mean`):

```toml
[tool.repro-lens.verify.stability]
accuracy = { max_std = 0.02, min_mean = 0.85 }
```

If any metric violates its stability bounds, `verify` exits with code 1 (`unstable`) and details the variance failure in the report and `$GITHUB_STEP_SUMMARY`. If all metrics pass, it exits with code 0 (`stable`).

