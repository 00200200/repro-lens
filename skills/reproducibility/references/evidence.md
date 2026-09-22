# Evidence and experiment choices

Distinguish a source-level risk, an observed run and a reproduced scientific claim.
Repro Lens implements screening and a limited two-run comparison.

An integer random_state resets an estimator's RNG between fits. A RandomState instance
can deliberately advance it, including between CV folds. Both can belong to a repeatable
overall program. New unseeded default_rng, SeedSequence or BitGenerator instances (`PCG64()`, …) do not inherit np.random.seed.
Seed evidence is needed at the random operation; `seed=42` in TOML alone proves nothing
about its use. Consult version-specific framework documentation before changing behavior.

Identify data revision, code including uncommitted changes, environment lock, command,
outputs and tolerances. Record blockers such as unavailable private data or a required
GPU. Do not substitute toy data and call the outcome reproduction of the original
experiment. A tiny fixture establishes only a smoke test.

Matching accuracy values are weak evidence if predictions differ. Prefer declared
predictions/split artifacts as well as metrics. Some formats contain timestamps;
compare a stable semantic export rather than ignoring a mismatch. Choose tolerances
from numerical expectations before looking at the outcome.

Primary references:
- [scikit-learn RNG semantics](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness)
- [NumPy generators](https://numpy.org/doc/stable/reference/random/generator.html)
- [PyTorch determinism](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
