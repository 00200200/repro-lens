"""Small, documented framework checks; no framework imports or project execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

RULES = {
    "R104": "XGBoost's gblinear booster selects the nondeterministic shotgun updater.",
    "R105": "LightGBM's deterministic configuration needs review.",
    "R106": "PyTorch data sampling has no explicit generator; review global RNG control.",
    "R107": "PyTorch cuDNN benchmarking is explicitly enabled.",
    "R108": "A TensorFlow generator is initialized from nondeterministic state.",
    "R109": "Lightning Trainer's deterministic configuration needs review.",
    "R110": "PyTorch deterministic algorithms are disabled or only warn on unsupported ops.",
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
    """Resolve only inline arguments, preserving uncertainty and dictionary overwrite order."""

    values: dict = field(default_factory=dict)
    unknown: bool = False

    def merge(self, other):
        if other.unknown:
            # An unknown dict expansion may replace any preceding dictionary key.
            self.values.clear()
            self.unknown = True
        self.values.update(other.values)

    @classmethod
    def mapping(cls, node):
        if not isinstance(node, ast.Dict):
            return cls(unknown=True)
        result = cls()
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                result.merge(cls.mapping(value))
            elif isinstance(constant(key), str):
                result.values[constant(key)] = value
            else:
                result.merge(cls(unknown=True))
        return result

    @classmethod
    def call(cls, node, positions=()):
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
                expanded = cls.mapping(keyword.value)
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


def check_data(node, name, emit):
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
    options = Arguments.call(node, positions)
    if name in LOADERS:
        shuffle = options.value("shuffle", False)
        if shuffle is False or shuffle is None:
            return
        if shuffle is not True:
            unresolved(node, name, emit)
            return
    generator = options.get("generator")
    if generator is UNKNOWN:
        unresolved(node, name, emit)
    elif generator is MISSING or constant(generator) is None:
        emit(
            node,
            "R106",
            f"{name} samples data without an explicit generator.",
            "Pass the experiment's seeded torch.Generator or review global torch RNG control. "
            "For DataLoader, also review random transforms and worker initialization.",
            "review",
        )


def check_trainer(node, name, emit):
    options = Arguments.call(node)
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


def check_algorithms(node, name, emit):
    options = Arguments.call(node, ("mode",))
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


def check_call(node, name, emit):
    if name in XGBOOST or name in {
        "xgboost.train",
        "xgboost.cv",
        "xgboost.training.train",
        "xgboost.training.cv",
    }:
        options = Arguments.call(node, ("params",))
        if name not in XGBOOST:
            options = Arguments.mapping(options.get("params"))
        check_xgboost(node, name, options, emit)
    elif name in LIGHTGBM or name in {
        "lightgbm.train",
        "lightgbm.cv",
        "lightgbm.engine.train",
        "lightgbm.engine.cv",
    }:
        options = Arguments.call(node, ("params",))
        if name not in LIGHTGBM:
            options = Arguments.mapping(options.get("params"))
        check_lightgbm(node, name, options, emit)
    elif name in LOADERS or name in SPLITS:
        check_data(node, name, emit)
    elif name in TF_NONDETERMINISTIC:
        emit(
            node,
            "R108",
            f"{name} explicitly initializes nondeterministic RNG state.",
            "Use Generator.from_seed with the experiment's seed, restore recorded state, "
            "or justify intentional nondeterminism.",
        )
    elif name in TRAINERS:
        check_trainer(node, name, emit)
    elif name == "torch.use_deterministic_algorithms":
        check_algorithms(node, name, emit)


def check_assignment(node, name, value, emit):
    if name == "torch.backends.cudnn.benchmark" and constant(value) is True:
        emit(
            node,
            "R107",
            "cuDNN benchmarking can select different algorithms across runs.",
            "Review benchmark=False for repeatability, or justify this performance choice. "
            "Deterministic kernels and RNG state require separate control.",
        )
