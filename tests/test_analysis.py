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


@pytest.mark.parametrize("name", ["PCG64", "PCG64DXSM", "MT19937", "SFC64", "Philox"])
def test_unseeded_numpy_bit_generators(name):
    imported = f"from numpy.random import {name}\n"
    assert codes(imported + f"{name}()") == ["R102"]
    assert codes(imported + f"{name}(None)") == ["R102"]
    assert codes(imported + f"{name}(seed=None)") == ["R102"]
    assert codes(imported + f"{name}(0)") == []
    assert codes(imported + f"{name}(seed)") == []
    assert codes(imported + f"{name}(**options)") == ["R190"]


def test_pcg64_inside_generator_is_the_unseeded_call():
    source = "from numpy.random import Generator, PCG64\nGenerator(PCG64())"
    active, _ = analyze(source, "train.py")
    column = source.splitlines()[1].index("PCG64()") + 1
    assert [(item.code, item.column) for item in active] == [("R102", column)]
    assert "numpy.random.PCG64" in active[0].message
    assert codes("from numpy.random import Generator, PCG64\nGenerator(PCG64(seed))") == []


def test_philox_key_pins_entropy_but_counter_does_not():
    imported = "from numpy.random import Philox\n"
    assert codes(imported + "Philox(key=1)") == []
    assert codes(imported + "Philox(key=stream)") == []
    assert codes(imported + "Philox(key=None)") == ["R102"]
    assert codes(imported + "Philox(counter=1)") == ["R102"]
    assert codes(imported + "Philox(seed=None, key=1)") == []


def test_mtrand_random_state_is_the_legacy_constructor():
    assert codes("from numpy.random.mtrand import RandomState\nRandomState()") == ["R102"]
    assert codes("from numpy.random.mtrand import RandomState\nRandomState(0)") == []


def test_unseeded_numpy_seed_sequence():
    imported = "from numpy.random import SeedSequence\n"
    assert codes(imported + "SeedSequence()") == ["R102"]
    assert codes(imported + "SeedSequence(None)") == ["R102"]
    assert codes(imported + "SeedSequence(entropy=None)") == ["R102"]
    # spawn_key alone still draws OS entropy for the pool.
    assert codes(imported + "SeedSequence(spawn_key=(1,))") == ["R102"]
    assert codes(imported + "SeedSequence(0)") == []
    assert codes(imported + "SeedSequence(entropy)") == []
    assert codes(imported + "SeedSequence(entropy=stream)") == []
    assert codes(imported + "SeedSequence(**options)") == ["R190"]
    assert codes("import numpy as np\nnp.random.SeedSequence()") == ["R102"]
    active, _ = analyze(imported + "SeedSequence()", "train.py")
    assert "entropy" in active[0].message


@pytest.mark.parametrize(
    "module, name",
    [
        ("sklearn.model_selection", "RandomizedSearchCV"),
        ("sklearn.linear_model", "RANSACRegressor"),
        ("sklearn.neural_network", "MLPClassifier"),
        ("sklearn.neural_network", "BernoulliRBM"),
        ("sklearn.cluster", "MiniBatchKMeans"),
        ("sklearn.cluster", "BisectingKMeans"),
        ("sklearn.mixture", "GaussianMixture"),
        ("sklearn.ensemble", "GradientBoostingRegressor"),
        ("sklearn.ensemble", "BaggingClassifier"),
        ("sklearn.ensemble", "IsolationForest"),
        ("sklearn.tree", "ExtraTreeClassifier"),
        ("sklearn.decomposition", "LatentDirichletAllocation"),
        ("sklearn.manifold", "TSNE"),
        ("sklearn.kernel_approximation", "Nystroem"),
        ("sklearn.random_projection", "SparseRandomProjection"),
        ("sklearn.inspection", "permutation_importance"),
        ("sklearn.utils", "shuffle"),
        ("sklearn.utils", "resample"),
    ],
)
def test_estimators_that_always_use_random_state(module, name):
    imported = f"from {module} import {name}\n"
    assert codes(imported + f"{name}()") == ["R101"]
    assert codes(imported + f"{name}(random_state=None)") == ["R101"]
    assert codes(imported + f"{name}(random_state=seed)") == []
    assert codes(imported + f"{name}(**options)") == ["R190"]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("from sklearn.linear_model import SGDClassifier\nSGDClassifier()", ["R101"]),
        ("from sklearn.linear_model import SGDRegressor\nSGDRegressor(shuffle=True)", ["R101"]),
        ("import sklearn.linear_model as lm\nlm.PassiveAggressiveRegressor()", ["R101"]),
        ("from sklearn.linear_model import SGDOneClassSVM\nSGDOneClassSVM(shuffle=False)", []),
        ("from sklearn.linear_model import SGDClassifier\nSGDClassifier(shuffle=flag)", ["R190"]),
        ("from sklearn.linear_model import SGDClassifier\nSGDClassifier(random_state=0)", []),
        ("from sklearn.model_selection import StratifiedGroupKFold as K\nK()", []),
        (
            "from sklearn.model_selection import StratifiedGroupKFold as K\nK(shuffle=True)",
            ["R101"],
        ),
        ("from sklearn.model_selection import learning_curve\nlearning_curve(m, X, y)", []),
        (
            "from sklearn.model_selection import learning_curve\n"
            "learning_curve(m, X, y, shuffle=True)",
            ["R101"],
        ),
        ("from sklearn.cluster import KMeans\nKMeans(init='random')", ["R101"]),
        ("from sklearn.cluster import KMeans\nKMeans(init=centroids)", ["R190"]),
        ("from sklearn.cluster import KMeans\nKMeans(init=centroids, random_state=0)", []),
        ("from sklearn.cluster import MiniBatchKMeans\nMiniBatchKMeans(init=centroids)", ["R101"]),
        # Seeded by default, or random only for some arguments: outside the registry.
        ("from sklearn.linear_model import Perceptron\nPerceptron()", []),
        ("from sklearn.model_selection import permutation_test_score as p\np(m, X, y)", []),
        ("from sklearn.decomposition import PCA\nPCA(svd_solver='randomized')", []),
        ("from sklearn.ensemble import HistGradientBoostingClassifier as H\nH()", []),
        ("import app\napp.KMeans()", []),
    ],
)
def test_shuffle_defaults_and_registry_limits(source, expected):
    assert codes(source) == expected


@pytest.mark.parametrize(
    "estimator",
    ["SGDClassifier", "SGDRegressor", "PassiveAggressiveClassifier", "PassiveAggressiveRegressor"],
)
@pytest.mark.parametrize(
    "arguments, expected",
    [
        ("shuffle=False", []),
        ("shuffle=False, early_stopping=False", []),
        ("shuffle=False, early_stopping=True", ["R101"]),
        ("shuffle=False, early_stopping=True, random_state=None", ["R101"]),
        ("shuffle=False, early_stopping=True, random_state=0", []),
        ("early_stopping=True", ["R101"]),
        ("shuffle=False, early_stopping=stop", ["R190"]),
        ("shuffle=False, early_stopping=stop, random_state=seed", []),
        ("shuffle=False, **options", ["R190"]),
        ("shuffle=False, early_stopping=True, **options", ["R190"]),
        ("shuffle=False, early_stopping=True, random_state=seed, **options", []),
    ],
)
def test_early_stopping_uses_random_state_without_shuffle(estimator, arguments, expected):
    source = f"from sklearn.linear_model import {estimator}\n{estimator}({arguments})"
    assert codes(source) == expected


def test_one_class_sgd_has_no_early_stopping_split():
    source = "from sklearn.linear_model import SGDOneClassSVM\nSGDOneClassSVM(shuffle=False)"
    assert codes(source) == []


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


@pytest.mark.parametrize(
    "nested",
    [
        "def transform(np):\n    return np.random.default_rng()",
        "async def transform(np):\n    return np.random.default_rng()",
        "def transform():\n    np = custom\n    return np.random.default_rng()",
        "class Transform:\n    np = custom\n    value = np.random.default_rng()",
        "transform = lambda np: np.random.default_rng()",
        "values = [np.random.default_rng() for np in objects]",
        "values = {np.random.default_rng() for np in objects}",
        "values = {np: np.random.default_rng() for np in objects}",
        "values = (np.random.default_rng() for np in objects)",
    ],
    ids=[
        "function",
        "async-function",
        "assignment",
        "class",
        "lambda",
        "list",
        "set",
        "dict",
        "generator",
    ],
)
@pytest.mark.parametrize("seed", ["", "1729"], ids=["unseeded", "seeded"])
def test_nested_scope_does_not_shadow_enclosing_import(nested, seed):
    source = (
        "import numpy as np\n\ndef train():\n"
        + textwrap.indent(nested, "    ")
        + f"\n    return np.random.default_rng({seed})\n"
    )
    active, _ = analyze(source, "train.py")
    assert [(item.code, item.line) for item in active] == (
        [("R102", len(source.splitlines()))] if not seed else []
    )


@pytest.mark.parametrize(
    "body",
    [
        "np = custom\n    def train(self):\n        return np.random.default_rng()",
        "np = np\n    def train(self):\n        return np.random.default_rng()",
        "def train(self):\n        return np.random.default_rng()\n    np = custom",
        (
            "np = custom\n    class Inner:\n        def train(self):\n"
            "            return np.random.default_rng()"
        ),
        "np = custom\n    train = lambda self: np.random.default_rng()",
        "np = custom\n    values = [np.random.default_rng() for i in items]",
        (
            "np = custom\n    def train(self, rng=np.random.default_rng()):\n"
            "        return np.random.default_rng()"
        ),
    ],
    ids=[
        "attribute",
        "alias",
        "attribute-after-method",
        "nested-class",
        "lambda",
        "comprehension",
        "method-default-and-body",
    ],
)
def test_class_body_names_do_not_hide_imports_in_methods(body):
    source = "import numpy as np\nclass Model:\n    " + body + "\n"
    active, _ = analyze(source, "train.py")
    assert [item.code for item in active] == ["R102"]
    assert all("default_rng" in item.message for item in active)


def test_class_body_still_uses_its_own_bindings():
    assert (
        codes("""
    import numpy as np
    class Model:
        np = custom
        value = np.random.default_rng()
        values = [i for i in np.random.default_rng()]
        def train(self, rng=np.random.default_rng()):
            return rng
    """)
        == []
    )


def test_method_closes_over_enclosing_function_not_class_attribute():
    assert (
        codes("""
    import numpy as np
    def factory():
        np = custom
        class Model:
            def train(self):
                return np.random.default_rng()
    """)
        == []
    )


def test_class_attribute_reexport_does_not_hide_method_sklearn_call():
    assert codes("""
    from sklearn.ensemble import RandomForestClassifier
    class Experiment:
        RandomForestClassifier = RandomForestClassifier
        def train(self):
            return RandomForestClassifier()
    """) == ["R101"]


@pytest.mark.parametrize(
    "binding",
    [
        "import custom as np",
        "from custom import np",
        "def np():\n    pass",
        "class np:\n    pass",
        "try:\n    work()\nexcept Exception as np:\n    pass",
        "match value:\n    case {'library': np}:\n        pass",
        "del np",
        "values = [(np := value) for value in objects]",
    ],
    ids=[
        "import",
        "from-import",
        "function",
        "class",
        "exception",
        "pattern",
        "deletion",
        "walrus",
    ],
)
def test_function_local_bindings_shadow_imports_before_assignment(binding):
    source = "import numpy as np\ndef train():\n    np.random.default_rng()\n" + textwrap.indent(
        binding, "    "
    )
    assert codes(source) == []


def test_local_import_still_resolves_inside_nested_function():
    assert codes("""
    import numpy as np
    def train(np):
        def transform():
            import numpy as np
            return np.random.default_rng()
        return np.random.default_rng()
    """) == ["R102"]


def test_global_declaration_preserves_import_until_rebinding():
    assert codes("""
    import numpy as np
    def train():
        global np
        np.random.default_rng()
        np = custom
        np.random.default_rng()
    """) == ["R102"]


def test_nonlocal_declaration_uses_enclosing_import():
    assert codes("""
    def train():
        import numpy as np
        def transform():
            nonlocal np
            np.random.default_rng()
            np = custom
            np.random.default_rng()
    """) == ["R102"]


def test_invalid_scope_declaration_is_not_a_clean_scan():
    active, suppressed = analyze("def train():\n    nonlocal missing\n", "train.py")
    assert [(item.code, item.line, item.severity) for item in active] == [("S902", 2, "error")]
    assert suppressed == []


@pytest.mark.parametrize(
    "name, default",
    [
        ("listcomp", "[np for np in objects]"),
        ("setcomp", "{np for np in objects}"),
        ("dictcomp", "{np: np for np in objects}"),
        ("genexpr", "(np for np in objects)"),
    ],
)
def test_default_comprehension_cannot_replace_function_scope(name, default):
    source = (
        f"import numpy as np\ndef {name}(values={default}):\n    return np.random.default_rng()\n"
    )
    assert codes(source) == ["R102"]


def test_function_scope_with_all_parameter_kinds():
    assert codes("""
    import numpy as np
    def train(a, /, b=1, *args, option=None, **kwargs):
        return np.random.default_rng()
    """) == ["R102"]


@pytest.mark.parametrize(
    "parameters",
    ["rng=random.Random({seed})", "rng=random.Random({seed}), /", "*, rng=random.Random({seed})"],
    ids=["positional", "positional-only", "keyword-only"],
)
@pytest.mark.parametrize("seed", ["", "1729"], ids=["unseeded", "seeded"])
def test_lambda_defaults_are_checked_in_the_enclosing_scope(parameters, seed):
    declaration = parameters.format(seed=seed)
    source = f"import random\nfactory = lambda {declaration}: rng\n"
    active, _ = analyze(source, "train.py")
    assert [(item.code, item.line) for item in active] == ([("R103", 2)] if not seed else [])


@pytest.mark.parametrize("seed", ["", "1729"], ids=["unseeded", "seeded"])
def test_parameter_of_a_lambda_default_does_not_shadow_the_outer_body(seed):
    source = (
        "import numpy as np\n"
        "factory = lambda callback=lambda np: np.random.default_rng(): "
        f"np.random.default_rng({seed})\n"
    )
    active, _ = analyze(source, "train.py")
    assert [(item.code, item.column) for item in active] == (
        [("R102", source.splitlines()[1].rindex("np.random") + 1)] if not seed else []
    )


@pytest.mark.parametrize("declaration", ["rng=rng()", "*, rng=rng()"])
def test_lambda_parameter_shadows_import_only_in_its_body(declaration):
    source = f"from numpy.random import default_rng as rng\nfactory = lambda {declaration}: rng()\n"
    active, _ = analyze(source, "train.py")
    assert [(item.code, item.column) for item in active] == [
        ("R102", source.splitlines()[1].index("rng()") + 1)
    ]


def test_lambda_defaults_keep_dynamic_arguments_and_suppressions():
    source = (
        "from sklearn.ensemble import RandomForestClassifier as Forest\n"
        "factory = lambda model=Forest(**options): model\n"
        "seeded = lambda model=Forest(random_state=config.seed): model\n"
        "shared = lambda model=Forest(): model  "
        "# repro-lens: ignore[R101] -- Shared RNG is seeded by the reviewed entrypoint.\n"
    )
    active, suppressed = analyze(source, "train.py")
    assert [(item.code, item.line, item.severity) for item in active] == [("R190", 2, "review")]
    assert [(item.code, item.line) for item in suppressed] == [("R101", 4)]


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


@pytest.mark.parametrize(
    "definition",
    [
        "@configure(rng())\ndef model(rng): pass",
        "@configure(rng())\nasync def model(rng): pass",
        "@configure(rng())\nclass Model: pass",
        "class Model(base(rng())): pass",
        "class Model(Base, random=rng()): pass",
        "class Model(Base, **options(rng())): pass",
        "@configure(rng())\ndef rng(): pass",
        "class rng(base(rng())): pass",
    ],
)
@pytest.mark.parametrize("argument, expected", [("", ["R102"]), ("1729", [])])
def test_definition_expressions_use_enclosing_imports(definition, argument, expected):
    source = "from numpy.random import default_rng as rng\n" + definition.replace(
        "rng()", f"rng({argument})", 1
    )
    assert codes(source) == expected


@pytest.mark.parametrize(
    "definition",
    [
        "@configure(rng())\ndef model(): pass",
        "@configure(rng())\nclass Model: pass",
        "class Model(base(rng())): pass",
        "class Model(Base, random=rng()): pass",
    ],
)
def test_definition_expressions_respect_shadowed_imports(definition):
    assert codes("from numpy.random import default_rng as rng\nrng = custom\n" + definition) == []


def test_decorator_finding_keeps_location_and_suppression():
    source = (
        "from numpy.random import default_rng as rng\n"
        "@configure(rng())  # repro-lens: ignore[R102] -- Intentional entropy.\n"
        "def model(): pass\n"
    )
    active, suppressed = analyze(source, "train.py")
    assert active == []
    assert [(item.code, item.line, item.column) for item in suppressed] == [("R102", 2, 12)]
