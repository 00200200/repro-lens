"""Conservative, import-aware checks. Never import or execute the scanned project."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    code: str
    severity: str
    message: str
    suggestion: str

    def to_dict(self):
        return asdict(self)


RULES = {
    "R101": "A known randomized scikit-learn call has no explicit random_state.",
    "R102": "A new NumPy generator is initialized without explicit entropy control.",
    "R103": "A new Python Random instance is initialized without explicit entropy control.",
    "R190": "Dynamic arguments prevent deciding whether randomness is controlled.",
    "P201": "A file required by the project's own policy is missing.",
    "P202": "Project policy or experiment configuration is invalid.",
    "P203": "A configured verification input is missing or escapes the project.",
    "S901": "A suppression needs a known rule and a nonempty justification.",
    "S902": "Python source could not be parsed; it was not checked.",
}

SKLEARN = {
    "sklearn.model_selection.train_test_split": "split",
    "sklearn.model_selection.KFold": "cv",
    "sklearn.model_selection.StratifiedKFold": "cv",
    "sklearn.model_selection.ShuffleSplit": "always",
    "sklearn.model_selection.StratifiedShuffleSplit": "always",
    "sklearn.model_selection.GroupShuffleSplit": "always",
    "sklearn.model_selection.RepeatedKFold": "always",
    "sklearn.model_selection.RepeatedStratifiedKFold": "always",
    "sklearn.ensemble.RandomForestClassifier": "always",
    "sklearn.ensemble.RandomForestRegressor": "always",
    "sklearn.ensemble.ExtraTreesClassifier": "always",
    "sklearn.ensemble.ExtraTreesRegressor": "always",
    "sklearn.tree.DecisionTreeClassifier": "always",
    "sklearn.tree.DecisionTreeRegressor": "always",
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
    def __init__(self, path):
        self.path = path
        self.bindings = {}
        self.findings = []

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
            for name in bound_names(target):
                self.bindings.pop(name, None)

    def visit_AnnAssign(self, node):
        if node.value:
            self.visit(node.value)
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
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default:
                self.visit(default)
        self.bindings.pop(node.name, None)
        outer = self.bindings
        self.bindings = outer.copy()
        # Python local names shadow globals throughout the function, even before assignment.
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                self.bindings.pop(child.id, None)
            elif isinstance(child, ast.arg):
                self.bindings.pop(child.arg, None)
        for statement in node.body:
            self.visit(statement)
        self.bindings = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bindings.pop(node.name, None)
        outer = self.bindings
        self.bindings = outer.copy()
        for statement in node.body:
            self.visit(statement)
        self.bindings = outer

    def visit_Lambda(self, node):
        outer = self.bindings
        self.bindings = outer.copy()
        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg):
                self.bindings.pop(argument.arg, None)
        self.visit(node.body)
        self.bindings = outer

    def visit_Call(self, node):
        name = self.qualified(node.func)
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        dynamic = any(kw.arg is None for kw in node.keywords) or any(
            isinstance(arg, ast.Starred) for arg in node.args
        )
        if name in SKLEARN:
            kind = SKLEARN[name]
            shuffle = literal(kwargs.get("shuffle"))
            if "shuffle" not in kwargs:
                shuffle = UNKNOWN if dynamic else kind == "split"
            if kind in {"split", "cv"} and shuffle is False:
                pass
            elif kind in {"split", "cv"} and shuffle is not True:
                self.emit(
                    node,
                    "R190",
                    f"Cannot resolve shuffle in {name}.",
                    "Review the effective shuffle and random_state values.",
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
    except (SyntaxError, ValueError) as exc:
        return [
            Finding(
                path,
                getattr(exc, "lineno", None) or 1,
                1,
                "S902",
                "error",
                str(exc),
                "Fix parsing before relying on this scan.",
            )
        ], []
    scanner = Scanner(path)
    scanner.visit(tree)
    suppressions = {}
    invalid = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or "repro-lens: ignore" not in token.string:
            continue
        match = re.fullmatch(
            r"#\s*repro-lens: ignore\[([A-Z0-9, ]+)\]\s*--\s*(\S.*)", token.string.strip()
        )
        codes = {code.strip() for code in match[1].split(",")} if match else set()
        if not codes or not codes <= {"R101", "R102", "R103", "R190"}:
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
