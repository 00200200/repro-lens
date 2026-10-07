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
        ("DataLoader(dataset, num_workers=4)", [("R116", "review")]),
        (
            "DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True)",
            [
                ("R116", "review"),
                ("R106", "review"),
            ],
        ),
        ("DataLoader(dataset, num_workers=1, worker_init_fn=None)", [("R116", "review")]),
        ("DataLoader(dataset, num_workers=n)", [("R116", "review")]),
        ("DataLoader(dataset, num_workers=workers, worker_init_fn=None)", [("R116", "review")]),
        (
            "DataLoader(dataset, 32, False, None, None, 4)",
            [("R116", "review")],
        ),
        ("DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)", []),
        ("DataLoader(dataset, num_workers=n, worker_init_fn=seed_worker)", []),
        ("DataLoader(dataset, num_workers=0)", []),
        ("DataLoader(dataset, num_workers=0, worker_init_fn=None)", []),
        ("DataLoader(dataset)", []),
        ("DataLoader(dataset, batch_size=32, shuffle=True, generator=rng)", []),
        ("DataLoader(dataset, num_workers=4, shuffle=False, **options)", []),
        ("DataLoader(dataset, shuffle=False, **options)", []),
        ("DataLoader(dataset, **{'num_workers': 4})", [("R116", "review")]),
        (
            "DataLoader(dataset, **{'num_workers': 4, 'worker_init_fn': seed_worker})",
            [],
        ),
        (
            "from torch.utils.data.dataloader import DataLoader as Loader\n"
            "Loader(dataset, num_workers=2)",
            [("R116", "review")],
        ),
    ],
)
def test_pytorch_dataloader_workers_need_worker_init_fn(source, expected):
    imports = "import torch\nfrom torch.utils.data import DataLoader\n"
    assert findings(imports + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("RandomSampler(dataset)", [("R106", "review")]),
        ("RandomSampler(dataset, generator=None)", [("R106", "review")]),
        ("RandomSampler(dataset, generator=torch.Generator())", [("R106", "review")]),
        (
            "RandomSampler(dataset, generator=torch.Generator().manual_seed(None))",
            [("R106", "review")],
        ),
        ("RandomSampler(dataset, generator=torch.Generator().manual_seed())", [("R106", "review")]),
        ("RandomSampler(dataset, generator=rng)", []),
        ("RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))", []),
        ("RandomSampler(dataset, generator=torch.Generator().manual_seed(*args))", []),
        ("RandomSampler(dataset, True, 8, rng)", []),
        ("RandomSampler(dataset, **options)", [("R190", "review")]),
        ("RandomSampler(*args)", [("R190", "review")]),
        ("WeightedRandomSampler(weights, 8)", [("R106", "review")]),
        ("WeightedRandomSampler(weights, num_samples=8, generator=rng)", []),
        ("WeightedRandomSampler(weights, 8, True, torch.Generator())", [("R106", "review")]),
        ("SubsetRandomSampler(indices)", [("R106", "review")]),
        ("SubsetRandomSampler(indices, generator=rng)", []),
        ("SubsetRandomSampler(indices, torch.Generator())", [("R106", "review")]),
        ("SequentialSampler(dataset)", []),
        (
            "from torch.utils.data.sampler import RandomSampler as Sampler\nSampler(dataset)",
            [("R106", "review")],
        ),
        ("DataLoader(dataset, sampler=RandomSampler(dataset))", [("R106", "review")]),
        ("DataLoader(dataset, sampler=SequentialSampler(dataset))", []),
        ("torch.manual_seed(17)\nRandomSampler(dataset)", [("R106", "review")]),
    ],
)
def test_pytorch_random_samplers_need_a_seeded_generator(source, expected):
    imports = (
        "import torch\n"
        "from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, "
        "SubsetRandomSampler, WeightedRandomSampler\n"
    )
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
        (
            "torch.use_deterministic_algorithms(True, warn_only=True)",
            [("R110", "review"), ("R120", "review")],
        ),
        ("torch.use_deterministic_algorithms(True)", [("R120", "review")]),
        ("torch.use_deterministic_algorithms(mode=True, warn_only=False)", [("R120", "review")]),
        ("torch.use_deterministic_algorithms(mode=enabled)", [("R190", "review")]),
        (
            "torch.use_deterministic_algorithms(True, **options)",
            [("R190", "review"), ("R120", "review")],
        ),
        ("torch.use_deterministic_algorithms(*args)", [("R190", "review")]),
    ],
)
def test_pytorch_algorithm_policy(source, expected):
    assert findings("import torch\n" + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("torch.use_deterministic_algorithms(True)", [("R120", "review")]),
        ("torch.use_deterministic_algorithms(mode=True)", [("R120", "review")]),
        (
            "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
            "torch.use_deterministic_algorithms(True)",
            [],
        ),
        (
            'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"\n'
            "torch.use_deterministic_algorithms(True)",
            [],
        ),
        (
            'os.putenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n'
            "torch.use_deterministic_algorithms(True)",
            [],
        ),
        (
            "torch.use_deterministic_algorithms(True)\n"
            'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"',
            [],
        ),
        (
            'os.environ["OTHER"] = ":4096:8"\ntorch.use_deterministic_algorithms(True)',
            [("R120", "review")],
        ),
        (
            'os.putenv("OTHER", ":4096:8")\ntorch.use_deterministic_algorithms(True)',
            [("R120", "review")],
        ),
        (
            "from os import environ\nenviron['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
            "torch.use_deterministic_algorithms(True)",
            [],
        ),
        (
            "from os import putenv\nputenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')\n"
            "torch.use_deterministic_algorithms(True)",
            [],
        ),
        (
            "from torch import use_deterministic_algorithms as enable\nenable(True)",
            [("R120", "review")],
        ),
        (
            "from torch import use_deterministic_algorithms as enable\n"
            'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\nenable(True)',
            [],
        ),
        (
            "torch.use_deterministic_algorithms(True, warn_only=True)",
            [("R110", "review"), ("R120", "review")],
        ),
        ("torch.use_deterministic_algorithms(False)", [("R110", "review")]),
        ("torch.use_deterministic_algorithms(enabled)", [("R190", "review")]),
    ],
)
def test_pytorch_cublas_workspace_with_deterministic_algorithms(source, expected):
    assert findings("import os\nimport torch\n" + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("torch.cuda.manual_seed(42)", [("R124", "review")]),
        ("torch.cuda.manual_seed(seed)", [("R124", "review")]),
        ("torch.cuda.manual_seed_all(42)", []),
        (
            "torch.cuda.manual_seed(42)\ntorch.cuda.manual_seed_all(42)",
            [],
        ),
        (
            "torch.cuda.manual_seed_all(42)\ntorch.cuda.manual_seed(42)",
            [],
        ),
        ("torch.manual_seed(42)", []),
        ("torch.cuda.manual_seed()", [("R124", "review")]),
        (
            "from torch.cuda import manual_seed\nmanual_seed(42)",
            [("R124", "review")],
        ),
        (
            "from torch.cuda import manual_seed as seed_gpu\nseed_gpu(42)",
            [("R124", "review")],
        ),
        (
            "from torch.cuda import manual_seed, manual_seed_all\n"
            "manual_seed(42)\nmanual_seed_all(42)",
            [],
        ),
        (
            "from torch.cuda import manual_seed as seed_one\n"
            "from torch.cuda import manual_seed_all as seed_all\n"
            "seed_one(42)\nseed_all(42)",
            [],
        ),
        (
            "import torch.cuda as cuda\ncuda.manual_seed(42)",
            [("R124", "review")],
        ),
        (
            "import torch.cuda as cuda\ncuda.manual_seed(42)\ncuda.manual_seed_all(42)",
            [],
        ),
        (
            "torch.cuda.manual_seed(42)\ntorch.cuda.manual_seed(7)",
            [("R124", "review"), ("R124", "review")],
        ),
    ],
)
def test_pytorch_cuda_manual_seed_without_manual_seed_all(source, expected):
    assert findings("import torch\n" + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("torch.bincount(x)", [("R130", "review")]),
        ("torch.Tensor.bincount(x)", [("R130", "review")]),
        ("torch.Tensor.index_add_(x, 0, idx, src)", [("R130", "review")]),
        ("x.index_add_(0, idx, src)", [("R130", "review")]),
        ("torch.index_add_(x, 0, idx, src)", [("R130", "review")]),
        ("torch.index_add(x, 0, idx, src)", [("R130", "review")]),
        ("torch.Tensor.scatter_add_(x, 0, idx, src)", [("R130", "review")]),
        ("x.scatter_add_(0, idx, src)", [("R130", "review")]),
        ("torch.scatter_add_(x, 0, idx, src)", [("R130", "review")]),
        ("torch.scatter_add(x, 0, idx, src)", [("R130", "review")]),
        ("torch.nn.functional.ctc_loss(log_probs, targets, in_len, tg_len)", [("R130", "review")]),
        ("torch.nn.CTCLoss()", [("R130", "review")]),
        ("torch.nn.functional.interpolate(x, mode='bilinear')", [("R130", "review")]),
        ("torch.nn.functional.interpolate(x, mode='bicubic')", [("R130", "review")]),
        ("torch.nn.functional.interpolate(x, (10, 10), None, 'bilinear')", [("R130", "review")]),
        ("torch.nn.functional.interpolate(x, mode='nearest')", []),
        ("torch.nn.functional.interpolate(x, size=10)", []),
        ("torch.nn.functional.interpolate(x, mode=dyn)", [("R190", "review")]),
        ("torch.nn.functional.interpolate(x, **options)", [("R190", "review")]),
        ("from torch import bincount\nbincount(x)", [("R130", "review")]),
        ("from torch.nn import functional as F\nF.ctc_loss(lp, t, il, tl)", [("R130", "review")]),
        (
            "from torch.nn import functional as F\nF.interpolate(x, mode='bilinear')",
            [("R130", "review")],
        ),
        (
            "from torch.nn.functional import interpolate\ninterpolate(x, mode='bicubic')",
            [("R130", "review")],
        ),
    ],
)
def test_pytorch_cuda_atomic_add(source, expected):
    assert findings("import torch\n" + source) == expected


def test_numpy_bincount_not_flagged_as_cuda_atomic_add():
    assert findings("import numpy as np\nnp.bincount(x)") == []


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
        ("torch.utils.data", "lib.RandomSampler(data)", "R106"),
        ("torch.utils.data", "lib.WeightedRandomSampler(weights, 8)", "R106"),
        ("torch.utils.data", "lib.SubsetRandomSampler(indices)", "R106"),
        ("torch.utils.data", "lib.DataLoader(data, num_workers=4)", "R116"),
        ("torch", "lib.backends.cudnn.benchmark = True", "R107"),
        ("tensorflow", "lib.random.Generator.from_non_deterministic_state()", "R108"),
        ("lightning.pytorch", "lib.Trainer()", "R109"),
        ("transformers", "lib.TrainingArguments(output_dir='./out')", "R117"),
        ("torch", "lib.use_deterministic_algorithms(False)", "R110"),
        ("torch", "lib.use_deterministic_algorithms(True)", "R120"),
        ("torch", "lib.cuda.manual_seed(42)", "R124"),
        ("torch", "lib.bincount(x)", "R130"),
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


PD = "import pandas as pd\n"


@pytest.mark.parametrize(
    "source, expected",
    [
        (PD + "df.sample(frac=0.2)", ["R115"]),
        (PD + "df.sample(n=5, replace=True)", ["R115"]),
        (PD + "df.groupby('g').sample(n=1)", ["R115"]),
        (PD + "pd.read_csv(path).sample(frac=1.0, ignore_index=True)", ["R115"]),
        (PD + "df.sample(frac=0.2, random_state=None)", ["R115"]),
        ("from pandas import DataFrame\ndf.sample(frac=0.5)", ["R115"]),
        (PD + "df.sample(frac=0.2, random_state=seed)", []),
        (PD + "df.sample(frac=0.2, random_state=rng)", []),
        (PD + "import numpy as np\nnp.random.seed(seed)\ndf.sample(frac=0.2)", []),
        (PD + "from lightning import seed_everything\nseed_everything(1)\ndf.sample(n=3)", []),
        (PD + "import torch\ntorch.manual_seed(1)\ndf.sample(n=3)", ["R115"]),
        ("df.sample(frac=0.2)", []),
        ("import polars as pl\ndf.sample(fraction=0.2, seed=None)", ["R118"]),
        (PD + "df.sample()", []),
        (PD + "df.sample(5)", []),
        (PD + "from sklearn.mixture import GaussianMixture\ngmm.sample(100)", []),
        (PD + "kde.sample(n_samples=44)", []),
        (PD + "rng.sample(population, 3)", []),
        (PD + "rng.sample(population, k=3)", []),
        (PD + "dist.sample((5,))", []),
        (PD + "df.sample(**options)", []),
        (PD + "df.sample(*args)", []),
        (PD + "import random\nrandom.sample(items, 3)", ["R112"]),
    ],
)
def test_pandas_sample_without_random_state_is_reviewed(source, expected):
    assert [item[0] for item in findings(source)] == expected


PL = "import polars as pl\n"


@pytest.mark.parametrize(
    "source, expected",
    [
        (PL + "df.sample(fraction=0.2)", ["R118"]),
        (PL + "df.sample(n=100)", ["R118"]),
        (PL + "df.sample(n=5, with_replacement=True)", ["R118"]),
        (PL + "df.sample(fraction=0.2, shuffle=True)", ["R118"]),
        (PL + "pl.read_csv(path).sample(fraction=1.0)", ["R118"]),
        (PL + "df.sample(fraction=0.2, seed=None)", ["R118"]),
        ("from polars import DataFrame\ndf.sample(fraction=0.5)", ["R118"]),
        (PL + "df.sample(n=100, seed=42)", []),
        (PL + "df.sample(fraction=0.2, seed=123)", []),
        ("df.sample(n=100)", []),
        (PL + "df.sample()", []),
        (PL + "df.sample(100)", []),
        (PL + "df.sample(**options)", []),
    ],
)
def test_polars_sample_without_seed_is_reviewed(source, expected):
    assert [item[0] for item in findings(source)] == expected


def test_polars_sample_findings_are_review_items_with_locations():
    active, suppressed = analyze(
        "import polars as pl\n\ntrain = frame.sample(fraction=0.8)\n"
        "subset = frame.sample(n=5)  # repro-lens: ignore[R118] -- Exploratory preview only.\n",
        "split.py",
    )
    assert [(f.code, f.severity, f.line, f.column) for f in active] == [("R118", "review", 3, 9)]
    assert (
        "without an explicit seed draws non-deterministic random subsamples in Polars"
        in active[0].message
    )
    assert [(f.code, f.line) for f in suppressed] == [("R118", 4)]


HF = "import datasets\n"


@pytest.mark.parametrize(
    "source, expected",
    [
        (HF + "dataset = dataset.shuffle()", ["R127"]),
        (HF + "dataset.shuffle()", ["R127"]),
        (HF + "dataset.shuffle(seed=None)", ["R127"]),
        (HF + "dataset.shuffle(None)", ["R127"]),
        (HF + "dataset.shuffle(buffer_size=1000)", ["R127"]),
        (HF + "dataset.shuffle(keep_in_memory=True)", ["R127"]),
        (HF + "dataset.shuffle(generator=None)", ["R127"]),
        (HF + "dataset.shuffle(buffer_size=1000, seed=None)", ["R127"]),
        (HF + "dataset.shuffle(None, None)", ["R127"]),
        ("from datasets import load_dataset\nds = load_dataset('imdb').shuffle()", ["R127"]),
        ("from datasets import Dataset\nds = Dataset.from_dict({}).shuffle()", ["R127"]),
        (HF + "dataset.shuffle(seed=42)", []),
        (HF + "dataset.shuffle(42)", []),
        (HF + "dataset.shuffle(seed=123, buffer_size=1000)", []),
        (HF + "dataset.shuffle(generator=gen)", []),
        (HF + "dataset.shuffle(None, gen)", []),
        ("dataset.shuffle()", []),
        (HF + "dataset.shuffle(**options)", []),
        (HF + "dataset.shuffle(*args)", []),
        (HF + "import random\nrandom.shuffle(items)", ["R112"]),
    ],
)
def test_hf_dataset_shuffle_without_seed_is_reviewed(source, expected):
    assert [item[0] for item in findings(source)] == expected


def test_hf_dataset_shuffle_findings_are_review_items_with_locations():
    active, suppressed = analyze(
        "import datasets\n\n"
        "train = dataset.shuffle()\n"
        "subset = dataset.shuffle()  # repro-lens: ignore[R127] -- Exploratory shuffling only.\n",
        "split.py",
    )
    assert [(f.code, f.severity, f.line, f.column) for f in active] == [("R127", "review", 3, 9)]
    assert (
        "without an explicit seed causes non-deterministic ordering in Hugging Face Datasets"
        in active[0].message
    )
    assert [(f.code, f.line) for f in suppressed] == [("R127", 4)]


DASK = "import dask.dataframe as dd\n"


@pytest.mark.parametrize(
    "source, expected",
    [
        (DASK + "df = ddf.sample(frac=0.5)", ["R129"]),
        (DASK + "ddf.sample(frac=0.5)", ["R129"]),
        (DASK + "ddf.sample(frac=0.5, random_state=None)", ["R129"]),
        (DASK + "ddf.sample(frac=0.5, seed=None)", ["R129"]),
        (DASK + "ddf.sample(frac=0.5, replace=True)", ["R129"]),
        (DASK + "ddf.sample(frac=0.5, random_state=42)", []),
        (DASK + "ddf.sample(frac=0.5, seed=42)", []),
        (DASK + "ddf.sample(100)", []),
        (DASK + "ddf.sample(**options)", []),
        (DASK + "ddf.shuffle(on='col')", ["R129"]),
        (DASK + "ddf.shuffle('col')", ["R129"]),
        (DASK + "ddf.shuffle(on='col', npartitions=4)", ["R129"]),
        (DASK + "ddf.shuffle(on='col', random_state=None)", ["R129"]),
        (DASK + "ddf.shuffle(on='col', seed=None)", ["R129"]),
        (DASK + "ddf.shuffle(on='col', shuffle='disk')", ["R129"]),
        (DASK + "ddf.shuffle(on='col', shuffle='tasks')", []),
        (DASK + "ddf.shuffle(on='col', random_state=42)", []),
        (DASK + "ddf.shuffle(on='col', seed=42)", []),
        (DASK + "ddf.shuffle(**options)", []),
        ("ddf.sample(frac=0.5)", []),
        ("ddf.shuffle(on='col')", []),
    ],
)
def test_dask_sample_and_shuffle_without_seed_is_reviewed(source, expected):
    assert [item[0] for item in findings(source)] == expected


def test_dask_findings_are_review_items_with_locations():
    active, suppressed = analyze(
        "import dask.dataframe as dd\n\n"
        "sampled = ddf.sample(frac=0.5)\n"
        "shuffled = ddf.shuffle(on='col')  # repro-lens: ignore[R129] -- Shuffled elsewhere.\n",
        "dask_job.py",
    )
    assert [(f.code, f.severity, f.line, f.column) for f in active] == [("R129", "review", 3, 11)]
    assert [(f.code, f.line) for f in suppressed] == [("R129", 4)]


def test_pandas_sample_findings_are_review_items_with_locations():
    active, suppressed = analyze(
        "import pandas as pd\n\ntrain = frame.sample(frac=0.8)\n"
        "subset = frame.sample(n=5)  # repro-lens: ignore[R115] -- Exploratory preview only.\n",
        "split.py",
    )
    assert [(f.code, f.severity, f.line, f.column) for f in active] == [("R115", "review", 3, 9)]
    assert "without random_state draws from NumPy's global RNG" in active[0].message
    assert [(f.code, f.line) for f in suppressed] == [("R115", 4)]


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
    assert len(report["cases"]) == 16
    assert all(case["expected_behavior"] for case in report["cases"])
    cases = json.loads(demo.with_name("cases.json").read_text())
    cases[0]["before"] = cases[0]["after"]
    altered = tmp_path / "cases.json"
    altered.write_text(json.dumps(cases))
    result = subprocess.run([*command, "--cases", str(altered)], text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(output.read_text())["cases"][0]["expected_behavior"] is False


@pytest.mark.parametrize(
    "source, expected",
    [
        ("cb.CatBoostClassifier()", [("R119", "review")]),
        ("cb.CatBoostRegressor(iterations=100)", [("R119", "review")]),
        ("cb.CatBoost()", [("R119", "review")]),
        ("cb.CatBoostClassifier(random_seed=42)", []),
        ("cb.CatBoostRegressor(random_state=17)", []),
        ("cb.CatBoost(random_seed=0)", []),
        ("cb.CatBoostClassifier(random_seed=seed_var)", [("R190", "review")]),
        ("cb.CatBoostClassifier(**options)", [("R190", "review")]),
        ("cb.CatBoostClassifier(**{'random_seed': 42})", []),
        ("cb.CatBoostRegressor(**{'iterations': 100})", [("R119", "review")]),
    ],
)
def test_catboost_seed_conditions(source, expected):
    assert findings("import catboost as cb\n" + source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("TrainingArguments(output_dir='./results')", [("R117", "review")]),
        (
            "TrainingArguments(output_dir='./results', per_device_train_batch_size=16)",
            [("R117", "review")],
        ),
        ("TrainingArguments(output_dir='./results', seed=42)", [("R117", "review")]),
        (
            "TrainingArguments(output_dir='./results', full_determinism=True)",
            [("R117", "review")],
        ),
        (
            "TrainingArguments(output_dir='./results', data_seed=42)",
            [("R117", "review")],
        ),
        (
            "TrainingArguments(output_dir='./results', full_determinism=False, data_seed=42)",
            [("R117", "review")],
        ),
        (
            "TrainingArguments(output_dir='./results', full_determinism=True, data_seed=None)",
            [("R117", "review")],
        ),
        (
            "TrainingArguments(output_dir='./results', seed=42, data_seed=42, "
            "full_determinism=True)",
            [],
        ),
        (
            "TrainingArguments(output_dir='./results', data_seed=seed, full_determinism=True)",
            [],
        ),
        (
            "TrainingArguments(output_dir='./results', full_determinism=enabled, data_seed=42)",
            [("R190", "review")],
        ),
        ("TrainingArguments(**options)", [("R190", "review")]),
        (
            "TrainingArguments(output_dir='./results', full_determinism=True, **options)",
            [("R190", "review")],
        ),
        (
            "TrainingArguments(**{'output_dir': './results', 'full_determinism': True, "
            "'data_seed': 42})",
            [],
        ),
        (
            "TrainingArguments(**{'output_dir': './results', 'data_seed': 42})",
            [("R117", "review")],
        ),
    ],
)
def test_hf_training_arguments_determinism(source, expected):
    assert findings("from transformers import TrainingArguments\n" + source) == expected


@pytest.mark.parametrize(
    "prefix, call",
    [
        ("from transformers import TrainingArguments\n", "TrainingArguments"),
        ("from transformers.training_args import TrainingArguments\n", "TrainingArguments"),
        ("import transformers\n", "transformers.TrainingArguments"),
    ],
)
def test_hf_training_arguments_import_paths(prefix, call):
    good = f"{prefix}{call}(output_dir='./results', seed=1, data_seed=1, full_determinism=True)\n"
    bad = f"{prefix}{call}(output_dir='./results')\n"
    assert findings(good) == []
    assert findings(bad) == [("R117", "review")]


VLLM = "from vllm import SamplingParams\n"
HF_GEN = "from transformers import GenerationConfig\n"


@pytest.mark.parametrize(
    "source, expected",
    [
        (VLLM + "params = SamplingParams(temperature=0.8)", [("R125", "warning")]),
        (VLLM + "params = SamplingParams(temperature=0.8, seed=42)", []),
        (VLLM + "params = SamplingParams(temperature=0.0)", []),
        (VLLM + "params = SamplingParams(temperature=0)", []),
        (VLLM + "params = SamplingParams(temperature=0.7, seed=None)", [("R125", "warning")]),
        (VLLM + "params = SamplingParams()", [("R125", "warning")]),
        (VLLM + "params = SamplingParams(temperature=dynamic_val)", [("R190", "review")]),
        (VLLM + "params = SamplingParams(seed=dynamic_seed)", [("R190", "review")]),
        ("import vllm\nparams = vllm.SamplingParams(temperature=0.9)", [("R125", "warning")]),
        ("import vllm\nparams = vllm.SamplingParams(temperature=0.9, seed=123)", []),
        (HF_GEN + "cfg = GenerationConfig(do_sample=True)", [("R125", "warning")]),
        (HF_GEN + "cfg = GenerationConfig(do_sample=True, seed=42)", []),
        (HF_GEN + "cfg = GenerationConfig(do_sample=False)", []),
        (HF_GEN + "cfg = GenerationConfig()", []),
        (HF_GEN + "cfg = GenerationConfig(do_sample=True, seed=None)", [("R125", "warning")]),
        ("model.generate(input_ids, do_sample=True)", [("R125", "warning")]),
        ("model.generate(input_ids, do_sample=True, seed=42)", []),
        ("model.generate(input_ids, do_sample=False)", []),
        ("model.generate(input_ids)", []),
    ],
)
def test_r125_llm_generation_determinism(source, expected):
    assert findings(source) == expected


def test_r125_suppression():
    code = (
        "from vllm import SamplingParams\n"
        "params = SamplingParams(temperature=0.7)  # repro-lens: ignore[R125] -- Exploratory run.\n"
    )
    active, suppressed = analyze(code, "test.py")
    assert active == []
    assert len(suppressed) == 1
    assert suppressed[0].code == "R125"


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            """
            import ray
            import torch

            @ray.remote
            def worker():
                torch.manual_seed(42)
            """,
            [("R123", "review")],
        ),
        (
            """
            import ray
            import numpy as np

            @ray.remote
            def worker():
                np.random.seed(123)
            """,
            [("R123", "review")],
        ),
        (
            """
            import ray
            import random

            @ray.remote
            def worker():
                random.seed(99)
            """,
            [("R123", "review")],
        ),
        (
            """
            import ray
            import torch

            @ray.remote(num_cpus=2)
            def worker():
                torch.manual_seed(42)
            """,
            [("R123", "review")],
        ),
        (
            """
            import ray
            import torch

            @ray.remote
            class WorkerActor:
                def __init__(self):
                    torch.manual_seed(42)
            """,
            [("R123", "review")],
        ),
        (
            """
            from ray.train.torch import TorchTrainer
            import torch

            def train_loop():
                torch.manual_seed(42)

            trainer = TorchTrainer(train_loop_per_worker=train_loop)
            """,
            [("R123", "review")],
        ),
        (
            """
            from ray.train.data_parallel_trainer import DataParallelTrainer
            import torch

            def train_loop():
                torch.manual_seed(42)

            trainer = DataParallelTrainer(train_loop_per_worker=train_loop)
            """,
            [("R123", "review")],
        ),
        (
            """
            from ray.train.torch import TorchTrainer
            import torch

            def train_loop():
                torch.manual_seed(42)

            trainer = TorchTrainer(train_loop)
            """,
            [("R123", "review")],
        ),
        (
            """
            from ray.train.torch import TorchTrainer
            import torch

            trainer = TorchTrainer(train_loop_per_worker=lambda: torch.manual_seed(42))
            """,
            [("R123", "review")],
        ),
        (
            """
            import ray
            import torch

            @ray.remote
            def worker():
                torch.manual_seed(42 + ray.train.get_context().get_world_rank())
            """,
            [],
        ),
        (
            """
            from ray.train.torch import TorchTrainer
            import ray
            import torch

            def train_loop():
                torch.manual_seed(base_seed + ray.train.get_context().get_world_rank())

            trainer = TorchTrainer(train_loop_per_worker=train_loop)
            """,
            [],
        ),
        (
            """
            from ray.train.torch import TorchTrainer
            import torch

            def train_loop(worker_id):
                torch.manual_seed(42 + worker_id)

            trainer = TorchTrainer(train_loop_per_worker=train_loop)
            """,
            [],
        ),
        (
            """
            from ray.train.torch import TorchTrainer
            import ray
            import torch

            def train_loop():
                rank = ray.train.get_context().get_world_rank()
                torch.manual_seed(42 + rank)

            trainer = TorchTrainer(train_loop_per_worker=train_loop)
            """,
            [],
        ),
        (
            """
            import torch

            def normal_train():
                torch.manual_seed(42)
            """,
            [],
        ),
        (
            """
            import numpy as np

            def regular_func():
                np.random.seed(42)
            """,
            [],
        ),
        (
            """
            import random

            def regular_func():
                random.seed(42)
            """,
            [],
        ),
    ],
)
def test_r123_ray_distributed_seeding(source, expected):
    assert findings(source) == expected


def test_r123_suppression():
    code = (
        "import ray\n"
        "import torch\n"
        "@ray.remote\n"
        "def worker():\n"
        "    torch.manual_seed(42)  # repro-lens: ignore[R123] -- Single-actor test worker.\n"
    )
    active, suppressed = analyze(code, "test.py")
    assert active == []
    assert len(suppressed) == 1
    assert suppressed[0].code == "R123"
