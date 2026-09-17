"""Behavioral checks use source strings: none of the ML frameworks is installed/imported."""

import json
import textwrap

import pytest

from repro_lens.analysis import analyze
from repro_lens.cli import main
from repro_lens.project import check


def findings(source):
    active, suppressed = analyze(textwrap.dedent(source), "experiment.py")
    assert suppressed == []
    return [(item.code, item.severity) for item in active]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("xgb.XGBClassifier(booster='gblinear')", [("R104", "warning")]),
        ("xgb.XGBClassifier(booster='gblinear', updater=None)", [("R104", "warning")]),
        ("xgb.train({'booster': 'gblinear', 'updater': None}, data)", [("R104", "warning")]),
        ("xgb.XGBRegressor(booster='gblinear', random_state=17)", [("R104", "warning")]),
        ("xgb.XGBRanker(booster='gblinear', updater='shotgun')", [("R104", "warning")]),
        ("xgb.XGBClassifier(booster='gblinear', updater='coord_descent')", []),
        ("xgb.XGBClassifier()", []),
        ("xgb.XGBRegressor(random_state=None)", []),
        ("xgb.XGBClassifier(booster='gbtree', updater=custom)", []),
        ("xgb.XGBClassifier(booster=choice)", [("R190", "review")]),
        ("xgb.XGBClassifier(booster='gblinear', updater=choice)", [("R190", "review")]),
        ("xgb.XGBClassifier(booster='gblinear', **options)", [("R190", "review")]),
        ("xgb.XGBClassifier(booster='gbtree', **options)", []),
        ("xgb.XGBClassifier(**{'booster': 'gblinear'})", [("R104", "warning")]),
        ("xgb.train({'booster': 'gblinear', 'seed': 17}, data)", [("R104", "warning")]),
        ("xgb.cv(params={'booster': 'gblinear'}, dtrain=data)", [("R104", "warning")]),
        ("xgb.train({'booster': 'gblinear', 'updater': 'coord_descent'}, data)", []),
        ("xgb.train({'booster': 'gbtree'}, data)", []),
        ("xgb.train(params, data)", [("R190", "review")]),
        ("xgb.train(*args)", [("R190", "review")]),
        ("xgb.train(**{'params': {'booster': 'gblinear'}, 'dtrain': data})", [("R104", "warning")]),
        ("xgb.train({'booster': 'gblinear', **overrides}, data)", [("R190", "review")]),
        (
            "xgb.train({**defaults, 'booster': 'gblinear', 'updater': 'shotgun'}, data)",
            [("R104", "warning")],
        ),
        ("xgb.train({'booster': 'gblinear', **{'booster': 'gbtree'}}, data)", []),
        ("xgb.train({'booster': 'gblinear', key: value}, data)", [("R190", "review")]),
        ("xgb.train({key: value, 'booster': 'gbtree'}, data)", []),
        (
            "xgb.train({'booster': 'gblinear', 'updater': 'shotgun', "
            "'updater': 'coord_descent'}, data)",
            [],
        ),
    ],
)
def test_xgboost_conditions_and_native_parameter_maps(source, expected):
    assert findings("import xgboost as xgb\n" + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("lgb.LGBMClassifier()", [("R105", "review")]),
        ("lgb.LGBMRegressor(random_state=17)", [("R105", "review")]),
        ("lgb.LGBMRanker(deterministic=False)", [("R105", "review")]),
        ("lgb.LGBMClassifier(deterministic=True)", [("R105", "review")]),
        ("lgb.LGBMClassifier(deterministic=True, force_col_wise=True)", []),
        ("lgb.LGBMClassifier(deterministic=True, force_row_wise=True)", []),
        (
            "lgb.LGBMClassifier(deterministic=True, force_col_wise=True, force_row_wise=True)",
            [("R105", "review")],
        ),
        (
            "lgb.LGBMClassifier(deterministic=True, force_col_wise=False, force_row_wise=False)",
            [("R105", "review")],
        ),
        (
            "lgb.LGBMClassifier(deterministic=True, device='gpu', force_col_wise=True)",
            [("R105", "review")],
        ),
        ("lgb.LGBMClassifier(deterministic=True, device_type='cuda')", [("R105", "review")]),
        (
            "lgb.LGBMClassifier(deterministic=True, device_type='cpu', device='gpu', "
            "force_col_wise=True)",
            [],
        ),
        ("lgb.LGBMClassifier(deterministic=enabled)", [("R190", "review")]),
        ("lgb.LGBMClassifier(deterministic=True, device=target)", [("R190", "review")]),
        ("lgb.LGBMClassifier(deterministic=True, force_col_wise=enabled)", [("R190", "review")]),
        (
            "lgb.LGBMClassifier(deterministic=True, force_col_wise=True, **options)",
            [("R190", "review")],
        ),
        ("lgb.train({'deterministic': True}, data)", [("R105", "review")]),
        ("lgb.cv(params={'deterministic': True, 'force_row_wise': True}, train_set=data)", []),
        ("lgb.train({'deterministic': True, **overrides}, data)", [("R190", "review")]),
        ("lgb.train(params, data)", [("R190", "review")]),
        ("lgb.train({'deterministic': True, **{'force_col_wise': True}}, data)", []),
    ],
)
def test_lightgbm_cpu_device_and_histogram_policy(source, expected):
    assert findings("import lightgbm as lgb\n" + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("DataLoader(dataset, shuffle=True)", [("R106", "review")]),
        ("DataLoader(dataset, 32, True)", [("R106", "review")]),
        ("DataLoader(dataset, shuffle=True, generator=None)", [("R106", "review")]),
        ("DataLoader(dataset, shuffle=True, generator=torch.Generator())", [("R106", "review")]),
        (
            "DataLoader(dataset, shuffle=True, generator=torch.Generator(device='cpu'))",
            [("R106", "review")],
        ),
        (
            "DataLoader(dataset, shuffle=True, generator=torch.Generator().manual_seed(None))",
            [("R106", "review")],
        ),
        (
            "DataLoader(dataset, shuffle=True, generator=torch.Generator().manual_seed())",
            [("R106", "review")],
        ),
        (
            "from torch import Generator\nDataLoader(dataset, shuffle=True, generator=Generator())",
            [("R106", "review")],
        ),
        ("DataLoader(dataset, shuffle=True, generator=rng)", []),
        ("DataLoader(dataset, shuffle=True, generator=torch.Generator().manual_seed(seed))", []),
        (
            "DataLoader(dataset, shuffle=True, generator=torch.Generator().manual_seed(*args))",
            [],
        ),
        ("DataLoader(dataset, shuffle=False)", []),
        ("DataLoader(dataset)", []),
        ("DataLoader(dataset, shuffle=None)", []),
        ("DataLoader(dataset, sampler=custom_sampler)", []),
        ("DataLoader(dataset, shuffle=choice)", [("R190", "review")]),
        ("DataLoader(dataset, shuffle=True, **options)", [("R190", "review")]),
        ("DataLoader(dataset, shuffle=False, **options)", []),
        ("DataLoader(dataset, **{'shuffle': True, 'generator': rng})", []),
        ("DataLoader(*args)", [("R190", "review")]),
        ("DataLoader(dataset, *args, shuffle=True)", [("R190", "review")]),
        (
            "DataLoader(dataset, 32, True, None, None, 0, None, False, False, 0, None, None, rng)",
            [],
        ),
        ("random_split(dataset, [8, 2])", [("R106", "review")]),
        ("random_split(dataset, [8, 2], rng)", []),
        ("random_split(dataset, [8, 2], torch.Generator())", [("R106", "review")]),
        ("random_split(dataset, [8, 2], None)", [("R106", "review")]),
        ("random_split(dataset, lengths=[8, 2], generator=rng)", []),
        ("random_split(*args, generator=rng)", []),
        ("random_split(dataset, [8, 2], *args)", [("R190", "review")]),
        ("torch.manual_seed(17)\nrandom_split(dataset, [8, 2])", [("R106", "review")]),
    ],
)
def test_pytorch_sampling_does_not_claim_missing_global_control(source, expected):
    imports = "import torch\nfrom torch.utils.data import DataLoader, random_split\n"
    assert findings(imports + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("torch.backends.cudnn.benchmark = True", [("R107", "warning")]),
        ("torch.backends.cudnn.benchmark: bool = True", [("R107", "warning")]),
        ("torch.backends.cudnn.benchmark = False", []),
        ("torch.backends.cudnn.benchmark = enabled", []),
        ("torch.backends.cudnn.deterministic = True", []),
        ("torch.use_deterministic_algorithms(False)", [("R110", "review")]),
        ("torch.use_deterministic_algorithms(mode=False)", [("R110", "review")]),
        ("torch.use_deterministic_algorithms(True, warn_only=True)", [("R110", "review")]),
        ("torch.use_deterministic_algorithms(True)", []),
        ("torch.use_deterministic_algorithms(mode=True, warn_only=False)", []),
        ("torch.use_deterministic_algorithms(mode=enabled)", [("R190", "review")]),
        ("torch.use_deterministic_algorithms(True, **options)", [("R190", "review")]),
        ("torch.use_deterministic_algorithms(*args)", [("R190", "review")]),
    ],
)
def test_pytorch_algorithm_policy(source, expected):
    assert findings("import torch\n" + source) == expected


@pytest.mark.parametrize(
    "namespace",
    ["random", "random.experimental", "compat.v1.random", "compat.v1.random.experimental"],
)
def test_tensorflow_nondeterministic_constructors(namespace):
    prefix = f"import tensorflow as tf\ntf.{namespace}.Generator."
    assert findings(prefix + "from_non_deterministic_state()") == [("R108", "warning")]
    assert findings(prefix + "from_non_deterministic_state(alg='philox')") == [("R108", "warning")]
    assert findings(prefix + "from_seed(experiment_seed)") == []
    assert findings(prefix + "from_state(recorded_state, alg='philox')") == []


@pytest.mark.parametrize(
    "namespace",
    [
        "lightning",
        "lightning.pytorch",
        "pytorch_lightning",
        "lightning.pytorch.trainer",
        "pytorch_lightning.trainer",
    ],
)
@pytest.mark.parametrize(
    "arguments, expected",
    [
        ("", [("R109", "review")]),
        ("deterministic=False", [("R109", "review")]),
        ("deterministic=None", [("R109", "review")]),
        ("deterministic='warn'", [("R109", "review")]),
        ("deterministic=True", []),
        ("deterministic=True, benchmark=False", []),
        ("deterministic=True, benchmark=True", [("R109", "review")]),
        ("deterministic=enabled", [("R190", "review")]),
        ("**options", [("R190", "review")]),
        ("**{'deterministic': True, 'benchmark': False}", []),
    ],
)
def test_lightning_namespaces_and_modes(namespace, arguments, expected):
    assert findings(f"from {namespace} import Trainer as T\nT({arguments})") == expected


@pytest.mark.parametrize(
    "module, call, code",
    [
        ("xgboost", "lib.XGBClassifier(booster='gblinear')", "R104"),
        ("lightgbm", "lib.LGBMClassifier()", "R105"),
        ("torch.utils.data", "lib.random_split(data, sizes)", "R106"),
        ("torch", "lib.backends.cudnn.benchmark = True", "R107"),
        ("tensorflow", "lib.random.Generator.from_non_deterministic_state()", "R108"),
        ("lightning.pytorch", "lib.Trainer()", "R109"),
        ("torch", "lib.use_deterministic_algorithms(False)", "R110"),
    ],
)
def test_framework_aliases_shadowing_and_justified_suppression(module, call, code):
    imported = f"import {module} as lib\n"
    assert [item[0] for item in findings(imported + call)] == [code]
    assert findings(imported + "lib = custom\n" + call) == []
    assert findings(imported + "def train(lib):\n    " + call) == []
    assert findings(imported + "def train():\n    " + call + "\n    lib = custom") == []
    assert findings("import unrelated as lib\n" + call) == []
    source = imported + call + f"  # repro-lens: ignore[{code}] -- Reviewed experiment policy."
    active, suppressed = analyze(source, "train.py")
    assert active == []
    assert [(f.code, f.line, f.path) for f in suppressed] == [(code, 2, "train.py")]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import numpy as np\nnp.random.shuffle(x)", ["R111"]),
        ("import numpy as np\nnp.random.seed(seed)\nnp.random.shuffle(x)", []),
        ("import numpy as np\nnp.random.shuffle(x)\nnp.random.seed(seed)", []),
        ("import numpy as np\nnp.random.seed()\nnp.random.shuffle(x)", ["R111"]),
        ("import numpy as np\nnp.random.seed(None)\nnp.random.shuffle(x)", ["R111"]),
        ("import numpy as np\nnp.random.seed(seed=None)\nnp.random.shuffle(x)", ["R111"]),
        ("import numpy as np\nnp.random.seed(0)\nnp.random.shuffle(x)", []),
        ("import numpy as np\nnp.random.seed(*args)\nnp.random.shuffle(x)", []),
        ("from numpy.random import seed, shuffle\nseed()\nshuffle(x)", ["R111"]),
        ("from numpy.random import randint\nrandint(3)", ["R111"]),
        ("from numpy.random import seed, randint\nseed(0)\nrandint(3)", []),
        ("import numpy as np\nrng = np.random.default_rng(seed)\nrng.shuffle(x)", []),
        ("import random\nrandom.shuffle(x)", ["R112"]),
        ("import random\nrandom.seed(seed)\nrandom.shuffle(x)", []),
        ("import random\nrandom.seed()\nrandom.shuffle(x)", ["R112"]),
        ("import random\nrandom.seed(None)\nrandom.shuffle(x)", ["R112"]),
        ("import random\nrandom.seed(a=None)\nrandom.shuffle(x)", ["R112"]),
        ("import random\nrandom.Random(seed).shuffle(x)", []),
        ("import torch\ntorch.randn(3)", ["R113"]),
        ("import torch\ntorch.manual_seed(seed)\ntorch.randn(3)", []),
        ("import torch\ntorch.manual_seed()\ntorch.randn(3)", ["R113"]),
        ("import torch\ntorch.manual_seed(None)\ntorch.randn(3)", ["R113"]),
        ("import torch\ntorch.cuda.manual_seed_all(seed)\ntorch.randn(3)", ["R113"]),
        ("import torch\ntorch.randn(3, generator=g)", []),
        ("import torch\ntorch.randn(3, generator=None)", ["R113"]),
        ("import torch\ntorch.randn(3, **options)", []),
        ("from torch.nn import init\ninit.kaiming_normal_(w)", ["R113"]),
        ("import torch\ntorch.manual_seed(seed)\ntorch.nn.init.xavier_uniform_(w)", []),
        ("import tensorflow as tf\ntf.random.normal([2])", ["R114"]),
        ("import tensorflow as tf\ntf.random.set_seed(seed)\ntf.random.normal([2])", []),
        ("import tensorflow as tf\ntf.random.normal([2], seed=seed)", []),
        ("import tensorflow as tf\ntf.random.stateless_normal([2], seed=[1, 2])", []),
        ("import keras\nimport numpy as np\nkeras.utils.set_random_seed(1)\nnp.random.rand()", []),
        (
            "import numpy as np\nimport torch\nfrom lightning import seed_everything\n"
            "seed_everything(seed)\nnp.random.rand()\ntorch.rand(2)",
            [],
        ),
        (
            "import numpy as np\nfrom lightning import seed_everything\n"
            "seed_everything()\nnp.random.rand()",
            ["R111"],
        ),
        (
            "import numpy as np\nfrom lightning import seed_everything\n"
            "seed_everything(None)\nnp.random.rand()",
            ["R111"],
        ),
        (
            "import numpy as np\nimport torch\nfrom transformers import set_seed\n"
            "set_seed(seed)\nnp.random.rand()\ntorch.rand(2)",
            [],
        ),
        ("import numpy as np\nimport torch\ntorch.manual_seed(seed)\nnp.random.rand()", ["R111"]),
        ("import numpy as np\nnp.random.default_rng()", ["R102"]),
        ("import numpy as np\nnp.random.rand()\nnp.random.default_rng(seed)", ["R111"]),
        ("import app\napp.random.shuffle(x)", []),
        ("def f(random):\n    random.shuffle(x)", []),
    ],
)
def test_global_rng_use_is_reviewed_when_the_file_never_seeds_it(source, expected):
    assert [item[0] for item in findings(source)] == expected


def test_global_rng_findings_are_review_items_with_locations():
    active, suppressed = analyze(
        "import numpy as np\n\ndef batch():\n    return np.random.permutation(10)\n"
        "\nnoise = np.random.normal(size=3)  # repro-lens: ignore[R111] -- Seeded by run.py.\n",
        "train.py",
    )
    assert [(f.code, f.severity, f.line, f.column) for f in active] == [("R111", "review", 4, 12)]
    assert "np.random.seed" not in active[0].message
    assert "numpy.random.permutation draws from NumPy's global RNG" in active[0].message
    assert [(f.code, f.line) for f in suppressed] == [("R111", 6)]


def test_direct_imports_and_lambda_defaults():
    assert findings(
        "from tensorflow.random import Generator as RNG\n"
        "factory = lambda rng=RNG.from_non_deterministic_state(): rng"
    ) == [("R108", "warning")]
    assert findings("from torch.backends import cudnn as backend\nbackend.benchmark = True") == [
        ("R107", "warning")
    ]
    assert findings(
        "from xgboost.sklearn import XGBRegressor as Model\nModel(booster='gblinear')"
    ) == [("R104", "warning")]


def test_cli_framework_findings_exclusions_and_no_execution(tmp_path, capsys):
    marker = tmp_path / "must-not-exist"
    (tmp_path / "train.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "import tensorflow as tf\ntf.random.Generator.from_non_deterministic_state()\n"
        "from lightning.pytorch import Trainer\nTrainer()\n"
    )
    assert main(["check", "--root", str(tmp_path), "--format", "json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [(f["code"], f["severity"]) for f in report["findings"]] == [
        ("R108", "warning"),
        ("R109", "review"),
    ]
    assert not marker.exists()
    assert main(["check", "--root", str(tmp_path), "--fail-on", "error"]) == 0
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens]\nexclude=["train.py"]\n')
    assert check(tmp_path)["findings"] == []


def test_framework_reviews_do_not_block_commits(tmp_path):
    (tmp_path / "train.py").write_text("from lightning.pytorch import Trainer\nTrainer()\n")
    assert main(["check", "--root", str(tmp_path)]) == 0


def test_framework_demo_checks_expected_results_and_detects_a_lost_warning(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    demo = root / "examples/framework_checks/demo.py"
    output = tmp_path / "report.json"
    command = [sys.executable, str(demo), "--output", str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert len(report["cases"]) == 15
    assert all(case["expected_behavior"] for case in report["cases"])
    cases = json.loads(demo.with_name("cases.json").read_text())
    cases[0]["before"] = cases[0]["after"]
    altered = tmp_path / "cases.json"
    altered.write_text(json.dumps(cases))
    result = subprocess.run([*command, "--cases", str(altered)], text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(output.read_text())["cases"][0]["expected_behavior"] is False
