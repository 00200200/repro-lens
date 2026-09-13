import textwrap

import pytest

from repro_lens.analysis import analyze


def codes(source):
    active, _ = analyze(textwrap.dedent(source), "train.py")
    return [item.code for item in active]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("params = {'booster': 'gblinear'}\nxgb.train(params, data)", ["R104"]),
        ("params: dict = {'booster': 'gblinear'}\nxgb.train(params, data)", ["R104"]),
        ("params = {'booster': 'gblinear'}; xgb.train(params, data)", ["R104"]),
        ("params = {'booster': 'gblinear'}\nxgb.cv(params=params, dtrain=data)", ["R104"]),
        (
            "params = {'booster': 'gblinear', 'seed': runtime_seed}\nxgb.train(params, data)",
            ["R104"],
        ),
        (
            "params = {'booster': 'gblinear', 'updater': 'coord_descent'}\nxgb.train(params, data)",
            [],
        ),
        ("params = {'booster': 'gbtree'}\nxgb.train(params, data)", []),
        ("params = {'booster': selected}\nxgb.train(params, data)", ["R190"]),
        ("params = {'booster': 'gblinear', **overrides}\nxgb.train(params, data)", ["R190"]),
        (
            "params = {**defaults, 'booster': 'gblinear', 'updater': 'shotgun'}\n"
            "xgb.train(params, data)",
            ["R104"],
        ),
        (
            "params = {'booster': 'gblinear'}\nwatchlist = [(data, 'train')]\nnum_round = 4\n"
            "model = xgb.train(params, data, num_round, watchlist)",
            ["R104"],
        ),
        ("options = {'booster': 'gblinear'}\nxgb.XGBClassifier(**options)", ["R104"]),
        (
            "options = {'booster': 'gblinear', 'updater': 'coord_descent'}\n"
            "xgb.XGBClassifier(**options)",
            [],
        ),
        (
            "def train():\n    params = {'booster': 'gblinear'}\n"
            "    return xgb.train(params, data)",
            ["R104"],
        ),
        (
            "async def train():\n    params = {'booster': 'gblinear'}\n"
            "    return xgb.train(params, data)",
            ["R104"],
        ),
        (
            "class Experiment:\n    def train(self):\n        params = {'booster': 'gblinear'}\n"
            "        return xgb.train(params, data)",
            ["R104"],
        ),
    ],
)
def test_named_boosting_parameters(source, expected):
    assert codes("import xgboost as xgb\n" + source) == expected


@pytest.mark.parametrize(
    "intervening",
    [
        "params['updater'] = 'coord_descent'",
        "params.update(updater='coord_descent')",
        "params.clear()",
        "alias = params",
        "mutate(params)",
        "stash = [params]",
        "params = {'booster': 'gbtree'}",
        "params |= {'updater': 'coord_descent'}",
        "del params",
        "if condition:\n    params['updater'] = 'coord_descent'",
        "if condition:\n    params = custom",
        "for params in configurations:\n    pass",
        "with manager() as params:\n    pass",
        "def mutate_later():\n    params['updater'] = 'coord_descent'",
        "def unrelated(params):\n    pass",
        "class Holder:\n    config = params",
        "def params():\n    pass",
        "class params:\n    pass",
        "import custom as params",
        "from custom import params",
        "from custom import *",
        "try:\n    work()\nexcept Exception as params:\n    pass",
        "match value:\n    case {'config': params}:\n        pass",
        "match value:\n    case {'config': _, **params}:\n        pass",
        "match value:\n    case [*params]:\n        pass",
        "values = [(params := config) for config in configurations]",
        "exec('params.clear()')",
        "namespace = globals()",
        "namespace = locals",
    ],
)
def test_mutation_escape_shadowing_and_reflection_remain_unresolved(intervening):
    assert codes(
        "import xgboost as xgb\nparams = {'booster': 'gblinear'}\n"
        + intervening
        + "\nxgb.train(params, data)"
    ) == ["R190"]


@pytest.mark.parametrize(
    "consumer",
    [
        "if condition:\n    xgb.train(params, data)",
        "for data in batches:\n    xgb.train(params, data)",
        "while condition:\n    xgb.train(params, data)",
        "try:\n    xgb.train(params, data)\nexcept Exception:\n    pass",
        "with manager():\n    xgb.train(params, data)",
        "models = [xgb.train(params, data) for data in batches]",
        "models = {xgb.train(params, data) for data in batches}",
        "models = {data: xgb.train(params, data) for data in batches}",
        "models = (xgb.train(params, data) for data in batches)",
        "train = lambda: xgb.train(params, data)",
        "def train():\n    return xgb.train(params, data)",
        "def train(model=xgb.train(params, data)):\n    return model",
        "class Trainer:\n    model = xgb.train(params, data)",
    ],
)
def test_deferred_repeated_and_conditional_consumers_are_not_inlined(consumer):
    assert codes("import xgboost as xgb\nparams = {'booster': 'gblinear'}\n" + consumer) == ["R190"]


def test_repeated_use_is_not_assumed_immutable_between_library_calls():
    assert codes("""
        import xgboost as xgb
        params = {'booster': 'gblinear'}
        xgb.train(params, first)
        xgb.train(params, second)
    """) == ["R190", "R190"]


@pytest.mark.parametrize(
    "source",
    [
        "xgb.train(params, data)\nparams = {'booster': 'gblinear'}",
        "if condition:\n    params = {'booster': 'gblinear'}\nxgb.train(params, data)",
        "params = alias = {'booster': 'gblinear'}\nxgb.train(params, data)",
        "params, other = {'booster': 'gblinear'}, None\nxgb.train(params, data)",
        "params = dict(booster='gblinear')\nxgb.train(params, data)",
        "params = {'booster': 'gblinear'}\nxgb.train(params, data)\nparams.clear()",
        "global params\nparams = {'booster': 'gblinear'}\nxgb.train(params, data)",
        "params = load_parameters()\nxgb.train(params, data)",
    ],
)
def test_unsupported_assignments_and_later_escape_remain_review(source):
    assert codes("import xgboost as xgb\n" + source) == ["R190"]


def test_nonlocal_assignment_is_not_treated_as_fresh_local_parameters():
    assert codes("""
        import xgboost as xgb
        def outer():
            params = None
            def train():
                nonlocal params
                params = {'booster': 'gblinear'}
                return xgb.train(params, data)
    """) == ["R190"]


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "import lightgbm as lgb\nparams = {'deterministic': True}\nlgb.train(params, data)",
            ["R105"],
        ),
        (
            "import lightgbm as lgb\nparams = {'deterministic': True, 'force_row_wise': True}\n"
            "lgb.cv(params=params, train_set=data)",
            [],
        ),
        (
            "from lightning.pytorch import Trainer\noptions = {'deterministic': True}\n"
            "Trainer(**options)",
            [],
        ),
        (
            "from lightning.pytorch import Trainer\noptions = {'deterministic': 'warn'}\n"
            "Trainer(**options)",
            ["R109"],
        ),
        (
            "from torch.utils.data import DataLoader\noptions = {'shuffle': True}\n"
            "DataLoader(data, **options)",
            ["R106"],
        ),
        (
            "from torch.utils.data import DataLoader\n"
            "options = {'shuffle': True, 'generator': rng}\n"
            "DataLoader(data, **options)",
            [],
        ),
        (
            "import torch\noptions = {'mode': True, 'warn_only': True}\n"
            "torch.use_deterministic_algorithms(**options)",
            ["R110"],
        ),
        ("import custom as xgb\nparams = {'booster': 'gblinear'}\nxgb.train(params, data)", []),
        (
            "import xgboost as xgb\nxgb = custom\nparams = {'booster': 'gblinear'}\n"
            "xgb.train(params, data)",
            [],
        ),
    ],
)
def test_framework_options_share_existing_import_aware_rules(source, expected):
    assert codes(source) == expected


def test_report_and_suppression_stay_at_call_site():
    source = (
        "import xgboost as xgb\nparams = {'booster': 'gblinear'}\n"
        "xgb.train(params, data)  # repro-lens: ignore[R104] -- Reviewed updater policy.\n"
    )
    active, suppressed = analyze(source, "experiment.py")
    assert active == []
    assert [(f.code, f.path, f.line, f.column) for f in suppressed] == [
        ("R104", "experiment.py", 3, 1)
    ]


def test_resolved_values_are_not_scanned_twice():
    assert codes("""
        import numpy as np
        import xgboost as xgb
        params = {'booster': 'gblinear', 'seed': np.random.default_rng()}
        xgb.train(params, data)
    """) == ["R102", "R104"]


def test_named_configuration_never_executes_values(tmp_path):
    from repro_lens.project import check

    marker = tmp_path / "executed"
    (tmp_path / "train.py").write_text(
        "import xgboost as xgb\nfrom pathlib import Path\n"
        f"params = {{'booster': 'gblinear', 'seed': Path({str(marker)!r}).touch()}}\n"
        "xgb.train(params, data)\n"
    )
    report = check(tmp_path)
    assert [(f["code"], f["line"]) for f in report["findings"]] == [("R104", 4)]
    assert not marker.exists()
