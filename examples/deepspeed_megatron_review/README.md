# Reproducible DeepSpeed and Megatron-LM distributed training

This guide explains the seed boundaries that matter when a training job uses
data, tensor and pipeline parallelism. It is a configuration and review guide,
not a claim that a multi-GPU run is reproducible without measuring the actual
job.

## The seed boundary

Use one explicit base seed for the job and derive rank-local streams according
to the parallel dimension:

| Stream | Data-parallel ranks | Tensor/pipeline ranks | Why |
| --- | --- | --- | --- |
| Data shuffling | Same logical seed | Same logical seed | Corresponding data-parallel replicas must consume the same sample order |
| Model initialization | Coordinated by the framework | Rank-local offsets are expected | Tensor and pipeline partitions must not accidentally share every random draw |
| Dropout / activation RNG | Framework RNG tracker | Framework RNG tracker | Checkpoint restore must recover the stream, not only the integer seed |
| DataLoader workers | Explicit `worker_init_fn` or framework helper | Explicit worker offsets | Forked workers otherwise inherit ambiguous RNG state |

Do not call `torch.manual_seed()` independently inside each rank after
DeepSpeed or Megatron has initialized its RNG tracker. That can silently replace
the framework's model-parallel streams. Set the base seed once, let the
framework derive model-parallel seeds, and record the parallel sizes in the
experiment configuration.

## Minimal configuration checklist

Record these values in the experiment manifest or verification inputs:

```text
base_seed=1234
data_parallel_size=2
tensor_parallel_size=2
pipeline_parallel_size=2
micro_batch_size=...
gradient_accumulation_steps=...
world_size=8
```

For PyTorch data loading, use a distributed sampler and call `set_epoch(epoch)`
exactly once before iterating each epoch. Without it, every epoch can repeat the
same shard order even when the initial seed is fixed.

For DeepSpeed, keep the launcher arguments, `deepspeed` configuration, and
model-parallel initialization code under version control. For Megatron-LM,
record the model-parallel seed and the tensor/pipeline sizes together; a seed
value without those sizes is not enough to reconstruct the RNG topology.

## Repro Lens workflow

First run the static check from the project root:

```bash
repro-lens check --root . --format text
```

Then define a verification contract that includes the launcher/configuration
files and a small deterministic artifact. Do not put a multi-node launcher
shell string in `command`; use an argv list or a reviewed wrapper script:

```toml
[tool.repro-lens.verify]
command = ["{python}", "train_smoke.py", "--output", "{output}"]
inputs = ["train_smoke.py", "ds_config.json", "megatron_config.json", "uv.lock"]
metrics = ["loss"]
artifacts = ["predictions.json"]
result = "result.json"
timeout = 600
atol = 1e-8
rtol = 1e-6
hash-inputs = true
```

Run the same contract twice on the same topology:

```bash
uv sync --locked
repro-lens verify --root . --format json
```

For scientific stability, use independent initialization seeds only when the
experiment explicitly supports them:

```bash
repro-lens verify --root . --seeds 1234,1235,1236 --format json
```

Compare reports from two code revisions with `repro-lens compare`. A matching
report means the declared outputs agreed under the recorded environment; it
does not prove that an unrecorded dataset, CUDA kernel, network collective, or
hardware failure could not affect the result.

## What to retain for a distributed reproduction

- commit and dirty-tree state;
- launcher command and environment allowlist;
- DeepSpeed/Megatron/PyTorch/CUDA/NCCL versions;
- world size and data/tensor/pipeline parallel sizes;
- checkpoint revision and shard manifest;
- sampler epoch policy and worker count;
- loss/metric output and a small prediction or checksum artifact;
- rank-0 and collective error logs.

The example intentionally contains no model weights, datasets, generated logs,
or GPU-dependent fixture. A real multi-node result must be validated on the
target topology and reported with its hardware and software boundaries.
