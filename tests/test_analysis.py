import textwrap

import pytest

from repro_lens.analysis import analyze


def codes(source):
    active, _ = analyze(textwrap.dedent(source), "train.py")
    return [item.code for item in active]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("from sklearn.model_selection import train_test_split as split\nsplit(X, y)", ["R101"]),
        ("import sklearn.model_selection as ms\nms.train_test_split(X, y)", ["R101"]),
        ("import sklearn.model_selection\nsklearn.model_selection.train_test_split(X)", ["R101"]),
        (
            "from sklearn.model_selection import train_test_split\n"
            "train_test_split(X, random_state=None)",
            ["R101"],
        ),
        (
            "from sklearn.model_selection import train_test_split\n"
            "train_test_split(X, random_state=0)",
            [],
        ),
        (
            "from sklearn.model_selection import train_test_split\n"
            "train_test_split(X, random_state=config.seed)",
            [],
        ),
        (
            "from sklearn.model_selection import train_test_split\n"
            "train_test_split(X, shuffle=False)",
            [],
        ),
        ("from sklearn.model_selection import KFold\nKFold()", []),
        ("from sklearn.model_selection import KFold\nKFold(shuffle=False)", []),
        ("from sklearn.model_selection import KFold\nKFold(shuffle=True)", ["R101"]),
        ("from sklearn.model_selection import KFold\nKFold(**options)", ["R190"]),
        ("from sklearn.model_selection import KFold\nKFold(shuffle=enabled)", ["R190"]),
        (
            "from sklearn.ensemble import RandomForestClassifier as Forest\nForest(**options)",
            ["R190"],
        ),
        (
            "from sklearn.ensemble import RandomForestClassifier as Forest\n"
            "Forest(random_state=seed, **options)",
            [],
        ),
        ("import numpy as np\nnp.random.default_rng()", ["R102"]),
        ("import numpy as np\nnp.random.seed(42)\nnp.random.default_rng()", ["R102"]),
        ("from numpy.random import default_rng as rng\nrng(None)", ["R102"]),
        ("from numpy.random import default_rng as rng\nrng(0)", []),
        ("from numpy.random import default_rng as rng\nrng(*args)", ["R190"]),
        ("from numpy.random import default_rng as rng\nrng(seed=existing_generator)", []),
        ("import random\nrandom.Random()", ["R103"]),
        ("import random\nrandom.Random(0)", []),
        ("import random\nrandom.Random(x=seed)", []),
        ("import secrets\nsecrets.token_bytes()", []),
        ("import app\napp.train_test_split(X)", []),
        ("def train_test_split(X): pass\ntrain_test_split(X)", []),
        ("import torch\ntorch.manual_seed(42)", []),
    ],
)
def test_randomness_conditions(source, expected):
    assert codes(source) == expected


@pytest.mark.parametrize(
    "body",
    [
        "np = custom\nnp.random.default_rng()",
        "def f(np):\n    np.random.default_rng()",
        "def f():\n    np.random.default_rng()\n    np = custom",
        "for np in objects:\n    np.random.default_rng()",
        "values = [np.random.default_rng() for np in objects]",
        "with context() as np:\n    np.random.default_rng()",
        "if condition:\n    np = custom\nnp.random.default_rng()",
    ],
)
def test_shadowed_names_are_not_library_calls(body):
    assert codes("import numpy as np\n" + body) == []


def test_function_import_does_not_leak_into_another_function():
    assert (
        codes("""
    def a():
        import numpy as np
        np.random.default_rng(1)
    def b():
        np.random.default_rng()
    """)
        == []
    )


def test_real_finding_retains_exact_location():
    active, _ = analyze(
        "import numpy as np\n\ndef f():\n    np.random.default_rng()\n", "src/train.py"
    )
    assert (active[0].path, active[0].line, active[0].column) == ("src/train.py", 4, 5)


def test_justified_suppression_is_auditable():
    source = (
        "import numpy as np\nnp.random.default_rng()  "
        "# repro-lens: ignore[R102] -- Entropy is intentionally logged.\n"
    )
    active, suppressed = analyze(source)
    assert active == []
    assert [f.code for f in suppressed] == ["R102"]


@pytest.mark.parametrize(
    "comment",
    [
        "# repro-lens: ignore[R102]",
        "# repro-lens: ignore[R102] --   ",
        "# repro-lens: ignore[R999] -- because",
    ],
)
def test_invalid_suppressions_do_not_hide_findings(comment):
    result = codes(f"import numpy as np\nnp.random.default_rng()  {comment}")
    assert "R102" in result and "S901" in result


def test_string_cannot_suppress_a_finding():
    assert codes(
        'import numpy as np\nnp.random.default_rng(); s = "# repro-lens: ignore[R102] -- reason"'
    ) == ["R102"]


def test_unparseable_file_is_not_a_clean_scan():
    assert codes("def f(:") == ["S902"]
