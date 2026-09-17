"""Conservative, import-aware checks. Never import or execute the scanned project."""

from __future__ import annotations

import ast
import io
import re
import symtable
import tokenize
from dataclasses import asdict, dataclass

from . import frameworks
from .parameters import ParameterDictionaries


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    code: str
    severity: str
    message: str
    suggestion: str
    cell: int | None = None  # 1-based notebook cell; line is then relative to that cell.

    def to_dict(self):
        data = asdict(self)
        if self.cell is None:
            del data["cell"]
        return data


RULES = {
    "R101": "A known randomized scikit-learn call has no explicit random_state.",
    "R102": "A new NumPy generator is initialized without explicit entropy control.",
    "R103": "A new Python Random instance is initialized without explicit entropy control.",
    **frameworks.RULES,
    "R190": "Dynamic arguments prevent deciding whether randomness is controlled.",
    "P201": "A file required by the project's own policy is missing.",
    "P202": "Project policy or experiment configuration is invalid.",
    "P203": "A configured verification input is missing or escapes the project.",
    "S901": "A suppression needs a known rule and a nonempty justification.",
    "S902": "Python source failed syntax or scope validation; it was not checked.",
}

# "split": shuffle defaults to True; "cv": shuffle defaults to False; "sgd": shuffle defaults
# to True and early_stopping to False, and either uses random_state; "init": random unless
# init is an explicit array; "always": random_state is used whatever the other arguments.
# APIs seeded by default (Perceptron, permutation_test_score) or random only for some
# arguments (PCA, TruncatedSVD) are omitted.
SKLEARN = {
    "sklearn.model_selection.train_test_split": "split",
    "sklearn.model_selection.KFold": "cv",
    "sklearn.model_selection.StratifiedKFold": "cv",
    "sklearn.model_selection.StratifiedGroupKFold": "cv",
    "sklearn.model_selection.learning_curve": "cv",
    "sklearn.model_selection.ShuffleSplit": "always",
    "sklearn.model_selection.StratifiedShuffleSplit": "always",
    "sklearn.model_selection.GroupShuffleSplit": "always",
    "sklearn.model_selection.RepeatedKFold": "always",
    "sklearn.model_selection.RepeatedStratifiedKFold": "always",
    "sklearn.model_selection.RandomizedSearchCV": "always",
    "sklearn.linear_model.SGDClassifier": "sgd",
    "sklearn.linear_model.SGDRegressor": "sgd",
    "sklearn.linear_model.SGDOneClassSVM": "split",
    "sklearn.linear_model.PassiveAggressiveClassifier": "sgd",
    "sklearn.linear_model.PassiveAggressiveRegressor": "sgd",
    "sklearn.linear_model.RANSACRegressor": "always",
    "sklearn.neural_network.MLPClassifier": "always",
    "sklearn.neural_network.MLPRegressor": "always",
    "sklearn.neural_network.BernoulliRBM": "always",
    "sklearn.cluster.KMeans": "init",
    "sklearn.cluster.MiniBatchKMeans": "always",
    "sklearn.cluster.BisectingKMeans": "always",
    "sklearn.mixture.GaussianMixture": "always",
    "sklearn.mixture.BayesianGaussianMixture": "always",
    "sklearn.ensemble.RandomForestClassifier": "always",
    "sklearn.ensemble.RandomForestRegressor": "always",
    "sklearn.ensemble.ExtraTreesClassifier": "always",
    "sklearn.ensemble.ExtraTreesRegressor": "always",
    "sklearn.ensemble.GradientBoostingClassifier": "always",
    "sklearn.ensemble.GradientBoostingRegressor": "always",
    "sklearn.ensemble.BaggingClassifier": "always",
    "sklearn.ensemble.BaggingRegressor": "always",
    "sklearn.ensemble.IsolationForest": "always",
    "sklearn.ensemble.RandomTreesEmbedding": "always",
    "sklearn.tree.DecisionTreeClassifier": "always",
    "sklearn.tree.DecisionTreeRegressor": "always",
    "sklearn.tree.ExtraTreeClassifier": "always",
    "sklearn.tree.ExtraTreeRegressor": "always",
    "sklearn.decomposition.LatentDirichletAllocation": "always",
    "sklearn.manifold.TSNE": "always",
    "sklearn.kernel_approximation.RBFSampler": "always",
    "sklearn.kernel_approximation.Nystroem": "always",
    "sklearn.random_projection.GaussianRandomProjection": "always",
    "sklearn.random_projection.SparseRandomProjection": "always",
    "sklearn.inspection.permutation_importance": "always",
    "sklearn.utils.shuffle": "always",
    "sklearn.utils.resample": "always",
    "sklearn.datasets.make_classification": "always",
    "sklearn.datasets.make_regression": "always",
    "sklearn.datasets.make_blobs": "always",
}
UNKNOWN = object()


def literal(node):
    if node is None:
        return UNKNOWN
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return UNKNOWN


def bound_names(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from bound_names(item)


class Scanner(ast.NodeVisitor):
    def __init__(self, path, symbols, tree):
        self.path = path
        self.bindings = {}
        self.findings = []
        self.global_uses = []
        self.seeded = set()
        self.parameters = ParameterDictionaries(tree)
        self.function_locals = {}
        pending = [symbols]
        while pending:
            scope = pending.pop()
            if isinstance(scope, symtable.Function):
                key = (scope.get_name(), scope.get_lineno(), frozenset(scope.get_parameters()))
                self.function_locals[key] = scope.get_locals()
            pending.extend(scope.get_children())

    def emit(self, node, code, message, suggestion, severity="warning"):
        self.findings.append(
            Finding(
                self.path, node.lineno, node.col_offset + 1, code, severity, message, suggestion
            )
        )

    def qualified(self, node):
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = self.qualified(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.bindings[name] = alias.name if alias.asname else name

    def visit_ImportFrom(self, node):
        if node.level or node.module is None:
            for alias in node.names:
                self.bindings.pop(alias.asname or alias.name, None)
            return
        for alias in node.names:
            if alias.name != "*":
                self.bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            frameworks.check_assignment(node, self.qualified(target), node.value, self.emit)
            for name in bound_names(target):
                self.bindings.pop(name, None)

    def visit_AnnAssign(self, node):
        if node.value:
            self.visit(node.value)
            frameworks.check_assignment(node, self.qualified(node.target), node.value, self.emit)
        for name in bound_names(node.target):
            self.bindings.pop(name, None)

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        for name in bound_names(node.target):
            self.bindings.pop(name, None)

    def visit_AugAssign(self, node):
        self.visit(node.value)
        for name in bound_names(node.target):
            self.bindings.pop(name, None)

    def visit_For(self, node):
        self.visit(node.iter)
        for name in bound_names(node.target):
            self.bindings.pop(name, None)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_If(self, node):
        self.visit(node.test)
        before = self.bindings.copy()
        for statement in node.body:
            self.visit(statement)
        then = self.bindings
        self.bindings = before.copy()
        for statement in node.orelse:
            self.visit(statement)
        self.bindings = {
            name: value for name, value in then.items() if self.bindings.get(name) == value
        }

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                for name in bound_names(item.optional_vars):
                    self.bindings.pop(name, None)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_ListComp(self, node):
        outer = self.bindings
        self.bindings = outer.copy()
        for generator in node.generators:
            self.visit(generator.iter)
            for name in bound_names(generator.target):
                self.bindings.pop(name, None)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.bindings = outer

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp
    visit_DictComp = visit_ListComp

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default:
                self.visit(default)
        self.bindings.pop(node.name, None)
        outer = self.bindings
        self.bindings = outer.copy()
        # The compiler distinguishes this function's locals from bindings in nested scopes.
        # Locals shadow outer imports throughout the function, even before assignment.
        parameters = frozenset(
            arg.arg for arg in ast.iter_child_nodes(node.args) if isinstance(arg, ast.arg)
        )
        # Parameters distinguish a function from a same-line comprehension in its defaults.
        for name in self.function_locals[node.name, node.lineno, parameters]:
            self.bindings.pop(name, None)
        for statement in node.body:
            self.visit(statement)
        self.bindings = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        # Header expressions run in the enclosing scope before the class is bound.
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        self.bindings.pop(node.name, None)
        outer = self.bindings
        self.bindings = outer.copy()
        for statement in node.body:
            self.visit(statement)
        self.bindings = outer

    def visit_Lambda(self, node):
        # Defaults are evaluated in the enclosing scope, before parameters shadow imports.
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        outer = self.bindings
        self.bindings = outer.copy()
        for argument in ast.iter_child_nodes(node.args):
            if isinstance(argument, ast.arg):
                self.bindings.pop(argument.arg, None)
        self.visit(node.body)
        self.bindings = outer

    def visit_Call(self, node):
        name = self.qualified(node.func)
        frameworks.check_call(node, name, self.emit, self.parameters.resolve, self.qualified)
        self.seeded |= frameworks.libraries_seeded(node, name)
        if library := frameworks.global_consumer(node, name):
            self.global_uses.append((node, name, library))
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        dynamic = any(kw.arg is None for kw in node.keywords) or any(
            isinstance(arg, ast.Starred) for arg in node.args
        )
        if name in SKLEARN:
            kind = SKLEARN[name]
            shuffle = literal(kwargs.get("shuffle"))
            if "shuffle" not in kwargs:
                shuffle = UNKNOWN if dynamic else kind in {"split", "sgd"}
            if kind == "sgd":
                # Early stopping draws a validation split from random_state, even without shuffle.
                early_stopping = literal(kwargs.get("early_stopping"))
                if "early_stopping" not in kwargs:
                    early_stopping = UNKNOWN if dynamic else False
                seed = kwargs.get("random_state")
                if shuffle is True or early_stopping is True:
                    self.check_seed(node, name, seed, dynamic, "R101")
                elif shuffle is False and early_stopping is False:
                    pass
                elif seed is None or literal(seed) is None:
                    self.emit(
                        node,
                        "R190",
                        f"Cannot resolve shuffle or early_stopping in {name}.",
                        "Review the effective shuffle, early_stopping and random_state values.",
                        "review",
                    )
            elif kind in {"split", "cv"} and shuffle is False:
                pass
            elif kind in {"split", "cv"} and shuffle is not True:
                self.emit(
                    node,
                    "R190",
                    f"Cannot resolve shuffle in {name}.",
                    "Review the effective shuffle and random_state values.",
                    "review",
                )
            elif (
                kind == "init" and "init" in kwargs and not isinstance(literal(kwargs["init"]), str)
            ):
                # Centroid arrays are deterministic; a callable or variable cannot be decided.
                seed = kwargs.get("random_state")
                if seed is None or literal(seed) is None:
                    self.emit(
                        node,
                        "R190",
                        f"Cannot resolve whether init in {name} uses random_state.",
                        "Review the effective init and random_state values.",
                        "review",
                    )
            else:
                self.check_seed(node, name, kwargs.get("random_state"), dynamic, "R101")
        elif name in {"numpy.random.default_rng", "numpy.random.RandomState", "random.Random"}:
            keyword = "x" if name == "random.Random" else "seed"
            seed = kwargs.get(keyword) or (node.args[0] if node.args else None)
            if isinstance(seed, ast.Starred):
                seed = None
            code = "R103" if name == "random.Random" else "R102"
            self.check_seed(node, name, seed, dynamic, code)
        self.generic_visit(node)

    def check_seed(self, node, name, seed, dynamic, code):
        if seed is not None and literal(seed) is not None:
            # An explicit expression is accepted, not proven to be a valid seeded RNG.
            return
        if dynamic and seed is None:
            self.emit(
                node,
                "R190",
                f"Cannot resolve expanded arguments in {name}.",
                "Inspect the supplied configuration; static analysis cannot decide.",
                "review",
            )
        else:
            parameter = "random_state" if code == "R101" else "seed"
            self.emit(
                node,
                code,
                f"{name} has no explicit non-None {parameter}.",
                "Pass the experiment's seed/RNG, or document intentional upstream RNG control.",
            )


def analyze(source: str, path: str = "<source>") -> tuple[list[Finding], list[Finding]]:
    try:
        tree = ast.parse(source, filename=path)
        symbols = symtable.symtable(source, path, "exec")
    except (SyntaxError, ValueError) as exc:
        return [
            Finding(
                path,
                getattr(exc, "lineno", None) or 1,
                1,
                "S902",
                "error",
                str(exc),
                "Fix syntax or scope declarations before relying on this scan.",
            )
        ], []
    scanner = Scanner(path, symbols, tree)
    scanner.visit(tree)
    frameworks.report_global_rng(scanner.global_uses, scanner.seeded, scanner.emit)
    suppressions = {}
    invalid = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or "repro-lens: ignore" not in token.string:
            continue
        match = re.fullmatch(
            r"#\s*repro-lens: ignore\[([A-Z0-9, ]+)\]\s*--\s*(\S.*)", token.string.strip()
        )
        codes = {code.strip() for code in match[1].split(",")} if match else set()
        if not codes or not codes <= {"R101", "R102", "R103", "R190", *frameworks.RULES}:
            invalid.append(
                Finding(
                    path,
                    token.start[0],
                    token.start[1] + 1,
                    "S901",
                    "error",
                    "Invalid or unexplained suppression.",
                    "Use '# repro-lens: ignore[R101] -- reason' on the call's first line.",
                )
            )
        else:
            suppressions[token.start[0]] = codes
    active, suppressed = [], []
    for finding in scanner.findings:
        target = suppressed if finding.code in suppressions.get(finding.line, set()) else active
        target.append(finding)
    return active + invalid, suppressed
