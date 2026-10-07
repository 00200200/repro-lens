"""Small, documented framework checks; no framework imports or project execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

RULES = {
    "R104": "XGBoost's gblinear booster selects the nondeterministic shotgun updater.",
    "R105": "LightGBM's deterministic configuration needs review.",
    "R106": "PyTorch data sampling has no explicit seeded generator; review global RNG control.",
    "R107": "PyTorch cuDNN benchmarking is explicitly enabled.",
    "R108": "A TensorFlow generator is initialized from nondeterministic state.",
    "R109": "Lightning Trainer's deterministic configuration needs review.",
    "R110": "PyTorch deterministic algorithms are disabled or only warn on unsupported ops.",
    "R111": "NumPy's global RNG is used, but this file never seeds it.",
    "R112": "Python's global random module is used, but this file never seeds it.",
    "R113": "PyTorch's global RNG is used, but this file never seeds it.",
    "R114": "TensorFlow's global RNG is used, but this file never seeds it.",
    "R115": "pandas sample() has no random_state, and this file never seeds NumPy's global RNG.",
    "R116": "PyTorch DataLoader uses multiple workers without worker_init_fn.",
    "R117": "Hugging Face TrainingArguments omits full_determinism=True or data_seed.",
    "R118": "Polars sample() has no explicit seed; review random subsampling.",
    "R120": "PyTorch deterministic algorithms are enabled without CUBLAS_WORKSPACE_CONFIG.",
    "R124": "torch.cuda.manual_seed seeds only the current GPU without manual_seed_all.",
    "R127": "Hugging Face dataset shuffle() has no explicit seed; review random shuffling.",
    "R129": "Dask DataFrame sample() or shuffle() has no explicit random_state or seed.",
    "R130": "PyTorch operation uses non-deterministic CUDA atomicAdd; review GPU reproducibility.",
}

MISSING = object()
UNKNOWN = object()


def constant(node):
    if node is MISSING or node is UNKNOWN:
        return node
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return UNKNOWN


@dataclass
class Arguments:
    """Resolve literal arguments and eligible named maps, preserving uncertainty and order."""

    values: dict = field(default_factory=dict)
    unknown: bool = False

    def merge(self, other):
        if other.unknown:
            # An unknown dict expansion may replace any preceding dictionary key.
            self.values.clear()
            self.unknown = True
        self.values.update(other.values)

    @classmethod
    def mapping(cls, node, resolve=None):
        if resolve is not None:
            node = resolve(node)
        if not isinstance(node, ast.Dict):
            return cls(unknown=True)
        result = cls()
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                result.merge(cls.mapping(value, resolve))
            elif isinstance(constant(key), str):
                result.values[constant(key)] = value
            else:
                result.merge(cls(unknown=True))
        return result

    @classmethod
    def call(cls, node, positions=(), resolve=None):
        result = cls()
        for index, argument in enumerate(node.args):
            if isinstance(argument, ast.Starred):
                for name in positions[index:]:
                    result.values[name] = UNKNOWN
                break
            if index < len(positions):
                result.values[positions[index]] = argument
        for keyword in node.keywords:
            if keyword.arg is None:
                expanded = cls.mapping(keyword.value, resolve)
                # Unlike dict displays, duplicate call keywords raise TypeError rather
                # than overwrite explicit arguments. Keep explicit values when unknown.
                result.unknown |= expanded.unknown
                result.values.update(expanded.values)
            else:
                result.values[keyword.arg] = keyword.value
        return result

    def get(self, name):
        return self.values.get(name, UNKNOWN if self.unknown else MISSING)

    def value(self, name, default=MISSING):
        value = constant(self.get(name))
        return default if value is MISSING else value


XGBOOST = {
    f"{module}.{estimator}"
    for module in ("xgboost", "xgboost.sklearn")
    for estimator in (
        "XGBModel",
        "XGBClassifier",
        "XGBRegressor",
        "XGBRanker",
        "XGBRFClassifier",
        "XGBRFRegressor",
    )
}
LIGHTGBM = {
    f"{module}.{estimator}"
    for module in ("lightgbm", "lightgbm.sklearn")
    for estimator in ("LGBMModel", "LGBMClassifier", "LGBMRegressor", "LGBMRanker")
}
CATBOOST = {
    f"{module}.{estimator}"
    for module in ("catboost",)
    for estimator in ("CatBoost", "CatBoostClassifier", "CatBoostRegressor")
}
TRAINERS = {
    f"{module}.Trainer"
    for module in (
        "lightning",
        "lightning.pytorch",
        "pytorch_lightning",
        "lightning.pytorch.trainer",
        "lightning.pytorch.trainer.trainer",
        "pytorch_lightning.trainer",
        "pytorch_lightning.trainer.trainer",
    )
}
HF_TRAINING_ARGS = {
    "transformers.TrainingArguments",
    "transformers.training_args.TrainingArguments",
}
TF_NONDETERMINISTIC = {
    f"tensorflow.{module}.Generator.from_non_deterministic_state"
    for module in (
        "random",
        "random.experimental",
        "compat.v1.random",
        "compat.v1.random.experimental",
    )
}
LOADERS = {"torch.utils.data.DataLoader", "torch.utils.data.dataloader.DataLoader"}
SPLITS = {"torch.utils.data.random_split", "torch.utils.data.dataset.random_split"}
SAMPLERS = {
    f"{module}.{name}": positions
    for module in ("torch.utils.data", "torch.utils.data.sampler")
    for name, positions in (
        ("RandomSampler", ("data_source", "replacement", "num_samples", "generator")),
        ("WeightedRandomSampler", ("weights", "num_samples", "replacement", "generator")),
        ("SubsetRandomSampler", ("indices", "generator")),
    )
}


# Global RNG state. Each library's seeder list follows its primary documentation:
# numpy.random.seed (legacy RandomState), random.seed, torch.manual_seed and
# tf.random.set_seed. Cross-library helpers seed several at once.
GLOBAL_RNG = {
    "numpy": {
        "code": "R111",
        "label": "NumPy's global RNG",
        "seeders": {"numpy.random.seed", "numpy.random.set_state"},
        "consumers": {
            f"numpy.random.{name}"
            for name in (
                "rand random randn random_sample ranf sample randint random_integers "
                "choice shuffle permutation bytes normal standard_normal uniform binomial "
                "poisson beta gamma exponential laplace lognormal geometric multinomial "
                "dirichlet multivariate_normal"
            ).split()
        },
        "hint": "numpy.random.seed(seed), or a numpy.random.default_rng(seed) Generator",
    },
    "python": {
        "code": "R112",
        "label": "Python's global random module",
        "seeders": {"random.seed", "random.setstate"},
        "consumers": {
            f"random.{name}"
            for name in (
                "random randint randrange choice choices sample shuffle uniform gauss "
                "normalvariate triangular betavariate expovariate getrandbits randbytes"
            ).split()
        },
        "hint": "random.seed(seed), or a random.Random(seed) instance",
    },
    "torch": {
        "code": "R113",
        "label": "PyTorch's global RNG",
        "seeders": {
            "torch.manual_seed",
            "torch.random.manual_seed",
            "torch.random.set_rng_state",
            "torch.set_rng_state",
        },
        "consumers": {
            f"torch.{name}"
            for name in (
                "rand randn randint randperm rand_like randn_like randint_like bernoulli "
                "multinomial normal poisson"
            ).split()
        }
        | {
            f"torch.nn.init.{name}"
            for name in (
                "uniform_ normal_ trunc_normal_ xavier_uniform_ xavier_normal_ "
                "kaiming_uniform_ kaiming_normal_ orthogonal_ sparse_"
            ).split()
        },
        "hint": "torch.manual_seed(seed), or pass generator=torch.Generator().manual_seed(seed)",
    },
    "tensorflow": {
        "code": "R114",
        "label": "TensorFlow's global RNG",
        "seeders": {
            "tensorflow.random.set_seed",
            "tensorflow.compat.v1.set_random_seed",
            "tensorflow.compat.v1.random.set_random_seed",
        },
        "consumers": {
            f"tensorflow.random.{name}"
            for name in "normal uniform truncated_normal shuffle categorical gamma poisson".split()
        },
        "hint": "tf.random.set_seed(seed), or a tf.random.Generator.from_seed(seed)",
    },
}
# Helpers that seed several libraries at once, per their documentation.
CROSS_SEEDERS = {
    **{
        f"{module}.seed_everything": {"python", "numpy", "torch"}
        for module in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning")
    },
    "transformers.set_seed": {"python", "numpy", "torch"},
    "transformers.trainer_utils.set_seed": {"python", "numpy", "torch"},
    "accelerate.utils.set_seed": {"python", "numpy", "torch"},
    "keras.utils.set_random_seed": {"python", "numpy", "tensorflow"},
    "tensorflow.keras.utils.set_random_seed": {"python", "numpy", "tensorflow"},
}
SEEDERS = {
    **{name: {library} for library, spec in GLOBAL_RNG.items() for name in spec["seeders"]},
    **CROSS_SEEDERS,
}
CONSUMERS = {name: library for library, spec in GLOBAL_RNG.items() for name in spec["consumers"]}
# First argument that actually pins entropy. seed()/seed(None) draw OS or clock entropy.
SEEDER_PARAMS = {
    **dict.fromkeys(SEEDERS, "seed"),
    "numpy.random.set_state": "state",
    "random.seed": "a",
    "random.setstate": "state",
    "torch.set_rng_state": "new_state",
    "torch.random.set_rng_state": "new_state",
}


def libraries_seeded(node, name):
    """Libraries this call seeds with an explicit non-None argument; otherwise none."""
    libraries = SEEDERS.get(name)
    if not libraries:
        return set()
    parameter = SEEDER_PARAMS[name]
    value = Arguments.call(node, (parameter,)).get(parameter)
    if value is MISSING or constant(value) is None:
        return set()
    return libraries


def global_consumer(node, name):
    """Return the library whose global RNG this call draws from, or None."""
    library = CONSUMERS.get(name)
    if library is None:
        return None
    # A torch generator= or TensorFlow seed= argument makes the call independent of
    # the global state (an operation seed alone gives a repeatable sequence).
    explicit = {"torch": "generator", "tensorflow": "seed"}.get(library)
    for keyword in node.keywords:
        if keyword.arg == explicit and constant(keyword.value) is not None:
            return None
        if keyword.arg is None:
            return None  # An expansion may carry that argument; stay silent.
    return library


# DataFrame, Series and GroupBy sample() accept these besides random_state (pandas 3.0).
PANDAS_SAMPLE_KEYWORDS = {"n", "frac", "replace", "weights", "axis", "ignore_index"}


def pandas_sample(node, name):
    """Return True for a pandas-style .sample() call that leaves random_state unset.

    The receiver's type is unknown, so only keyword calls shaped like pandas' signature
    count. Positional calls are left alone: gmm.sample(100) and kde.sample(44) in
    scikit-learn look the same as df.sample(5). So are imported functions such as
    random.sample and ** expansions.
    """
    func = node.func
    if name is not None or not isinstance(func, ast.Attribute) or func.attr != "sample":
        return False
    if node.args:
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords or not set(keywords) <= PANDAS_SAMPLE_KEYWORDS | {"random_state"}:
        return False
    if "random_state" in keywords:
        return constant(keywords["random_state"]) is None
    return bool(keywords)


def report_pandas_sample(calls, seeded, emit):
    """pandas draws from numpy.random when random_state is None, so a NumPy seed controls it."""
    if "numpy" in seeded:
        return
    for node in calls:
        emit(
            node,
            "R115",
            "sample() without random_state draws from NumPy's global RNG in pandas, "
            "which this file never seeds.",
            "Pass random_state=seed, or seed NumPy's global RNG in the reviewed entrypoint. "
            "The receiver is assumed to be a pandas object because this file imports pandas.",
            "review",
        )


# Polars DataFrame and LazyFrame sample() accept these besides seed.
POLARS_SAMPLE_KEYWORDS = {"n", "fraction", "with_replacement", "shuffle"}


def polars_sample(node, name):
    """Return True for a Polars-style .sample() call that leaves seed unset.

    The receiver's type is unknown, so only keyword calls shaped like Polars' signature
    count. Positional calls are left alone.
    """
    func = node.func
    if name is not None or not isinstance(func, ast.Attribute) or func.attr != "sample":
        return False
    if node.args:
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords or not set(keywords) <= POLARS_SAMPLE_KEYWORDS | {"seed"}:
        return False
    if "seed" in keywords:
        return constant(keywords["seed"]) is None
    return bool(keywords)


def report_polars_sample(calls, emit):
    """Emit one review item per Polars-style .sample() call without an explicit seed."""
    for node in calls:
        emit(
            node,
            "R118",
            "sample() without an explicit seed draws non-deterministic random subsamples in "
            "Polars.",
            "Pass seed=integer to ensure reproducible sampling in Polars pipelines. "
            "The receiver is assumed to be a Polars object because this file imports polars.",
            "review",
        )


HF_DATASET_SHUFFLE_KEYWORDS = {
    "seed",
    "generator",
    "keep_in_memory",
    "load_from_cache_file",
    "indices_cache_file_name",
    "writer_batch_size",
    "buffer_size",
    "num_proc",
}


def hf_dataset_shuffle(node, name):
    """Return True for a Hugging Face dataset .shuffle() call that leaves seed unset.

    The receiver's type is unknown, so calls shaped like Hugging Face Datasets'
    shuffle signature count when the file imports datasets.
    """
    func = node.func
    if name is not None or not isinstance(func, ast.Attribute) or func.attr != "shuffle":
        return False
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords:
        return False
    if not set(keywords) <= HF_DATASET_SHUFFLE_KEYWORDS:
        return False
    if node.args:
        if len(node.args) > 6:
            return False
        if len(node.args) >= 2 and constant(node.args[1]) is not None:
            return False
        return constant(node.args[0]) is None
    if "generator" in keywords and constant(keywords["generator"]) is not None:
        return False
    if "seed" in keywords:
        return constant(keywords["seed"]) is None
    return True


def report_hf_dataset_shuffle(calls, emit):
    """Emit one review item per Hugging Face dataset .shuffle() call without an explicit seed."""
    for node in calls:
        emit(
            node,
            "R127",
            "shuffle() without an explicit seed causes non-deterministic ordering in Hugging Face "
            "Datasets.",
            "Pass seed=integer to ensure reproducible shuffling and enable Hugging Face's disk "
            "caching. The receiver is assumed to be a dataset object because this file imports "
            "datasets.",
            "review",
        )


DASK_SAMPLE_KEYWORDS = {"frac", "replace", "random_state", "ignore_index", "seed"}


def dask_sample(node, name):
    """Return True for a Dask DataFrame .sample() call that leaves random_state unset."""
    func = node.func
    if name is not None or not isinstance(func, ast.Attribute) or func.attr != "sample":
        return False
    if node.args:
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords or not set(keywords) <= DASK_SAMPLE_KEYWORDS:
        return False
    if "random_state" in keywords and constant(keywords["random_state"]) is not None:
        return False
    if "seed" in keywords and constant(keywords["seed"]) is not None:
        return False
    return bool(keywords)


def report_dask_sample(calls, emit):
    """Emit one review item per Dask DataFrame .sample() call without an explicit seed."""
    for node in calls:
        emit(
            node,
            "R129",
            "sample() without an explicit random_state or seed draws non-deterministic "
            "partition samples in Dask.",
            "Pass random_state=integer or seed=integer to ensure reproducible partition "
            "sampling in Dask DataFrames. The receiver is assumed to be a Dask object "
            "because this file imports dask.",
            "review",
        )


DASK_SHUFFLE_KEYWORDS = {
    "on",
    "ignore_index",
    "npartitions",
    "max_branch",
    "shuffle",
    "compute",
    "random_state",
    "seed",
}


def dask_shuffle(node, name):
    """Return True for a Dask DataFrame .shuffle() call lacking deterministic configuration."""
    func = node.func
    if name is not None or not isinstance(func, ast.Attribute) or func.attr != "shuffle":
        return False
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return False
    if len(node.args) > 3:
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords or not set(keywords) <= DASK_SHUFFLE_KEYWORDS:
        return False
    if "random_state" in keywords and constant(keywords["random_state"]) is not None:
        return False
    if "seed" in keywords and constant(keywords["seed"]) is not None:
        return False
    if "shuffle" in keywords and constant(keywords["shuffle"]) == "tasks":
        return False
    return True


def report_dask_shuffle(calls, emit):
    """Emit review items per Dask DataFrame .shuffle() call lacking determinism."""
    for node in calls:
        emit(
            node,
            "R129",
            "shuffle() without explicit random_state, seed, or shuffle='tasks' causes "
            "non-deterministic partition ordering in Dask.",
            "Specify shuffle='tasks' or pass random_state=integer to ensure deterministic "
            "partition division across scheduler runs. The receiver is assumed to be a "
            "Dask object because this file imports dask.",
            "review",
        )


def report_global_rng(uses, seeded, emit):
    """Emit one review item per global RNG use in a file that never seeds that library."""
    for node, name, library in uses:
        if library in seeded:
            continue
        spec = GLOBAL_RNG[library]
        emit(
            node,
            spec["code"],
            f"{name} draws from {spec['label']}, which this file never seeds.",
            f"Seed it in the reviewed entrypoint, for example {spec['hint']}, "
            "or justify seeding that happens elsewhere.",
            "review",
        )


def unresolved(node, name, emit):
    emit(
        node,
        "R190",
        f"Cannot resolve reproducibility options in {name}.",
        "Inspect the effective configuration; no project code was evaluated.",
        "review",
    )


def check_xgboost(node, name, options, emit):
    booster = options.value("booster", "gbtree")
    if booster is UNKNOWN:
        unresolved(node, name, emit)
    elif booster == "gblinear":
        updater = options.value("updater", "shotgun")
        # XGBoost's Python bindings skip None-valued parameters, leaving the default.
        if updater is None or updater == "shotgun":
            emit(
                node,
                "R104",
                f"{name} selects gblinear with the nondeterministic shotgun updater.",
                "Review coord_descent for this experiment, or justify shotgun; a seed alone "
                "does not remove its parallel update nondeterminism.",
            )
        elif updater is UNKNOWN:
            unresolved(node, name, emit)


def check_catboost(node, name, options, emit):
    seed = options.value("random_seed")
    if seed is MISSING:
        seed = options.value("random_state")

    if seed is UNKNOWN:
        unresolved(node, name, emit)
    elif seed is MISSING:
        emit(
            node,
            "R119",
            f"{name} initialized without explicit random_seed or random_state.",
            "Pass an explicit random_seed (or random_state) to ensure reproducible training.",
            "review",
        )


def check_lightgbm(node, name, options, emit):
    deterministic = options.value("deterministic", False)
    if deterministic is UNKNOWN:
        unresolved(node, name, emit)
        return
    if deterministic is not True:
        emit(
            node,
            "R105",
            f"{name} does not explicitly enable CPU deterministic mode.",
            "Review deterministic=True for CPU training, or document the actual device and "
            "repeatability policy; default seeds can already be repeatable.",
            "review",
        )
        return
    # LightGBM gives the primary device_type key precedence over its device alias.
    device = options.value("device_type")
    if device is MISSING:
        device = options.value("device", "cpu")
    if device is UNKNOWN:
        unresolved(node, name, emit)
    elif device != "cpu":
        emit(
            node,
            "R105",
            f"{name} enables deterministic=True for a non-CPU device.",
            "LightGBM's deterministic setting applies only to CPU training; review "
            "device-specific repeatability and verify the declared outputs.",
            "review",
        )
    else:
        col = options.value("force_col_wise", False)
        row = options.value("force_row_wise", False)
        if col is UNKNOWN or row is UNKNOWN:
            unresolved(node, name, emit)
        elif (col is True) == (row is True):
            emit(
                node,
                "R105",
                f"{name} enables determinism without exactly one forced histogram mode.",
                "Review force_col_wise=True or force_row_wise=True (choose one), as recommended "
                "by LightGBM for deterministic CPU training.",
                "review",
            )


def unseeded_torch_generator(node, qualified):
    """True for torch.Generator() or Generator().manual_seed() / manual_seed(None)."""
    if qualified is None or not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed":
        ctor = node.func.value
        if not (isinstance(ctor, ast.Call) and qualified(ctor.func) == "torch.Generator"):
            return False
        seed = Arguments.call(node, ("seed",)).get("seed")
        if seed is UNKNOWN:
            return False  # *args / **kwargs may carry a seed
        return seed is MISSING or constant(seed) is None
    return qualified(node.func) == "torch.Generator"


def check_generator(node, name, options, emit, qualified=None):
    generator = options.get("generator")
    if generator is UNKNOWN:
        unresolved(node, name, emit)
    elif (
        generator is MISSING
        or constant(generator) is None
        or unseeded_torch_generator(generator, qualified)
    ):
        emit(
            node,
            "R106",
            f"{name} samples data without an explicit seeded generator.",
            "Pass torch.Generator().manual_seed with the experiment's seed, or review global "
            "torch RNG control. Also review random transforms and DataLoader worker "
            "initialization.",
            "review",
        )


def check_dataloader_workers(node, name, options, emit):
    """Review multi-worker DataLoaders that omit worker_init_fn (PyTorch reproducibility)."""
    workers = options.get("num_workers")
    # Omitted num_workers defaults to 0. An unknown ** expansion alone is not enough
    # evidence that workers are active.
    if workers is MISSING or workers is UNKNOWN:
        return
    workers_value = constant(workers)
    positive_literal = (
        isinstance(workers_value, int) and not isinstance(workers_value, bool) and workers_value > 0
    )
    if workers_value is not UNKNOWN and not positive_literal:
        return
    init_fn = options.get("worker_init_fn")
    if init_fn is UNKNOWN:
        return  # ** expansion may carry worker_init_fn.
    if init_fn is not MISSING and constant(init_fn) is not None:
        return
    emit(
        node,
        "R116",
        f"{name} uses num_workers > 0 without worker_init_fn; workers inherit the "
        "parent process RNG state.",
        "Pass a worker_init_fn that seeds NumPy and Python RNGs per worker, or use "
        "num_workers=0. See PyTorch's DataLoader reproducibility notes.",
        "review",
    )


def check_data(node, name, emit, resolve, qualified=None):
    positions = (
        ("dataset", "lengths", "generator")
        if name in SPLITS
        else (
            "dataset",
            "batch_size",
            "shuffle",
            "sampler",
            "batch_sampler",
            "num_workers",
            "collate_fn",
            "pin_memory",
            "drop_last",
            "timeout",
            "worker_init_fn",
            "multiprocessing_context",
            "generator",
        )
    )
    options = Arguments.call(node, positions, resolve)
    if name in LOADERS:
        check_dataloader_workers(node, name, options, emit)
        shuffle = options.value("shuffle", False)
        if shuffle is False or shuffle is None:
            return
        if shuffle is not True:
            unresolved(node, name, emit)
            return
    check_generator(node, name, options, emit, qualified)


def check_trainer(node, name, emit, resolve):
    options = Arguments.call(node, resolve=resolve)
    deterministic = options.value("deterministic", None)
    benchmark = options.value("benchmark", None)
    if deterministic is UNKNOWN or benchmark is UNKNOWN:
        unresolved(node, name, emit)
    elif deterministic is not True or benchmark is True:
        emit(
            node,
            "R109",
            f"{name} does not request strict determinism with benchmarking disabled.",
            "Review deterministic=True, benchmark=False and seed_everything(seed, workers=True); "
            "external setup may already control determinism. The 'warn' mode is not strict.",
            "review",
        )


def check_hf_training_args(node, name, emit, resolve):
    """Review TrainingArguments when full_determinism is not True or data_seed is unset/None."""
    options = Arguments.call(node, resolve=resolve)
    full_determinism = options.get("full_determinism")
    data_seed = options.get("data_seed")
    # ** expansions without an explicit key leave UNKNOWN; do not guess.
    if full_determinism is UNKNOWN or data_seed is UNKNOWN:
        unresolved(node, name, emit)
        return
    full_value = constant(full_determinism)
    if full_determinism is not MISSING and full_value is UNKNOWN:
        unresolved(node, name, emit)
        return
    data_value = constant(data_seed)
    # Non-literal data_seed expressions are accepted without evaluating them.
    needs_determinism = full_value is not True
    needs_data_seed = data_seed is MISSING or data_value is None
    if needs_determinism or needs_data_seed:
        emit(
            node,
            "R117",
            f"{name} omits full_determinism=True or an explicit non-None data_seed.",
            "Pass full_determinism=True and data_seed=<integer> (and usually seed=) so "
            "transformers enables deterministic algorithms and pins data shuffling.",
            "review",
        )


def check_algorithms(node, name, emit, resolve):
    options = Arguments.call(node, ("mode",), resolve)
    mode = options.value("mode")
    warn = options.value("warn_only", False)
    if mode is UNKNOWN or warn is UNKNOWN:
        unresolved(node, name, emit)
    elif mode is False or (mode is True and warn is True):
        emit(
            node,
            "R110",
            f"{name} allows operations without a deterministic implementation.",
            "Review strict mode (mode=True, warn_only=False) or justify the allowed operations "
            "and verify their outputs in the target environment.",
            "review",
        )


CUBLAS_WORKSPACE = "CUBLAS_WORKSPACE_CONFIG"


def sets_cublas_workspace(target, qualified):
    """True when assigning to os.environ['CUBLAS_WORKSPACE_CONFIG'] (any os.environ alias)."""
    if not isinstance(target, ast.Subscript):
        return False
    if qualified(target.value) != "os.environ":
        return False
    return constant(target.slice) == CUBLAS_WORKSPACE


def puts_cublas_workspace(node, name):
    """True for os.putenv('CUBLAS_WORKSPACE_CONFIG', ...) (any os.putenv alias)."""
    if name != "os.putenv" or not node.args:
        return False
    return constant(node.args[0]) == CUBLAS_WORKSPACE


def requests_deterministic_algorithms(node, name, resolve=None):
    """True when torch.use_deterministic_algorithms is called with literal True mode."""
    if name != "torch.use_deterministic_algorithms":
        return False
    return Arguments.call(node, ("mode",), resolve).value("mode") is True


def report_cublas_workspace(calls, configured, emit):
    """Review deterministic-algorithm enables when CUBLAS_WORKSPACE_CONFIG is unset in-file."""
    if configured:
        return
    for node, name in calls:
        emit(
            node,
            "R120",
            f"{name} enables deterministic algorithms without setting "
            f"{CUBLAS_WORKSPACE} in this file.",
            f'Set os.environ["{CUBLAS_WORKSPACE}"] to ":4096:8" or ":16:8" before CUDA '
            "initialization, or justify CPU-only / non-CUDA use.",
            "review",
        )


def is_cuda_manual_seed(name):
    """True for torch.cuda.manual_seed (any imported alias), not manual_seed_all."""
    return name == "torch.cuda.manual_seed"


def is_cuda_manual_seed_all(name):
    """True for torch.cuda.manual_seed_all (any imported alias)."""
    return name == "torch.cuda.manual_seed_all"


def report_cuda_manual_seed(calls, seeded_all, emit):
    """Review cuda.manual_seed when the same file never calls cuda.manual_seed_all."""
    if seeded_all:
        return
    for node, name in calls:
        emit(
            node,
            "R124",
            f"{name} seeds only the current CUDA device RNG.",
            "Prefer torch.cuda.manual_seed_all(seed) so every visible GPU is seeded, "
            "or justify single-GPU affinity (for example CUDA_VISIBLE_DEVICES).",
            "review",
        )


TORCH_ATOMIC_ADD_NAMES = {
    "torch.bincount",
    "torch.Tensor.bincount",
    "torch.Tensor.index_add_",
    "torch.Tensor.scatter_add_",
    "torch.Tensor.index_add",
    "torch.Tensor.scatter_add",
    "torch.index_add",
    "torch.index_add_",
    "torch.scatter_add",
    "torch.scatter_add_",
    "torch.nn.functional.ctc_loss",
    "torch.nn.CTCLoss",
}

TORCH_ATOMIC_ADD_ATTRS = {
    "index_add_",
    "scatter_add_",
    "index_add",
    "scatter_add",
}


def check_torch_interpolate(node, name, emit, resolve):
    options = Arguments.call(node, ("input", "size", "scale_factor", "mode"), resolve)
    mode = options.value("mode", "nearest")
    if mode is UNKNOWN:
        unresolved(node, name, emit)
    elif isinstance(mode, str) and mode.lower() in {"bilinear", "bicubic"}:
        emit(
            node,
            "R130",
            f"{name} with mode={mode!r} uses non-deterministic CUDA atomicAdd on GPU.",
            "CUDA floating-point atomicAdd introduces non-deterministic accumulation order due to "
            "thread race conditions. Consider torch.use_deterministic_algorithms(True) or CPU "
            "execution for exact reproducibility.",
            "review",
        )


def check_torch_atomic_add(node, name, emit):
    target = None
    if name in TORCH_ATOMIC_ADD_NAMES:
        target = name
    elif (
        name is None
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in TORCH_ATOMIC_ADD_ATTRS
    ):
        target = f"{node.func.attr}()"

    if target is not None:
        emit(
            node,
            "R130",
            f"{target} uses non-deterministic CUDA atomicAdd on GPU.",
            "CUDA floating-point atomicAdd introduces non-deterministic accumulation order due to "
            "thread race conditions. Consider torch.use_deterministic_algorithms(True) or CPU "
            "execution for exact reproducibility.",
            "review",
        )
        return True
    return False


def check_call(node, name, emit, resolve=None, qualified=None):
    if name in XGBOOST or name in {
        "xgboost.train",
        "xgboost.cv",
        "xgboost.training.train",
        "xgboost.training.cv",
    }:
        options = Arguments.call(node, ("params",), resolve)
        if name not in XGBOOST:
            options = Arguments.mapping(options.get("params"), resolve)
        check_xgboost(node, name, options, emit)
    elif name in LIGHTGBM or name in {
        "lightgbm.train",
        "lightgbm.cv",
        "lightgbm.engine.train",
        "lightgbm.engine.cv",
    }:
        options = Arguments.call(node, ("params",), resolve)
        if name not in LIGHTGBM:
            options = Arguments.mapping(options.get("params"), resolve)
        check_lightgbm(node, name, options, emit)
    elif name in CATBOOST:
        options = Arguments.call(node, (), resolve)
        check_catboost(node, name, options, emit)
    elif name in LOADERS or name in SPLITS:
        check_data(node, name, emit, resolve, qualified)
    elif name in SAMPLERS:
        check_generator(node, name, Arguments.call(node, SAMPLERS[name], resolve), emit, qualified)
    elif name in TF_NONDETERMINISTIC:
        emit(
            node,
            "R108",
            f"{name} explicitly initializes nondeterministic RNG state.",
            "Use Generator.from_seed with the experiment's seed, restore recorded state, "
            "or justify intentional nondeterminism.",
        )
    elif name in TRAINERS:
        check_trainer(node, name, emit, resolve)
    elif name in HF_TRAINING_ARGS:
        check_hf_training_args(node, name, emit, resolve)
    elif name == "torch.use_deterministic_algorithms":
        check_algorithms(node, name, emit, resolve)
    elif name == "torch.nn.functional.interpolate":
        check_torch_interpolate(node, name, emit, resolve)
    elif check_torch_atomic_add(node, name, emit):
        pass


def check_assignment(node, name, value, emit):
    if name == "torch.backends.cudnn.benchmark" and constant(value) is True:
        emit(
            node,
            "R107",
            "cuDNN benchmarking can select different algorithms across runs.",
            "Review benchmark=False for repeatability, or justify this performance choice. "
            "Deterministic kernels and RNG state require separate control.",
        )
