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
    elif name == "torch.use_deterministic_algorithms":
        check_algorithms(node, name, emit, resolve)


def check_assignment(node, name, value, emit):
    if name == "torch.backends.cudnn.benchmark" and constant(value) is True:
        emit(
            node,
            "R107",
            "cuDNN benchmarking can select different algorithms across runs.",
            "Review benchmark=False for repeatability, or justify this performance choice. "
            "Deterministic kernels and RNG state require separate control.",
        )
