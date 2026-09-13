"""Recognize a deliberately narrow, non-escaping use of a named dictionary."""

import ast
from collections import defaultdict


def other_bindings(node):
    """Bindings/declarations whose identifiers are not represented by ast.Name."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return node.names
    if isinstance(node, ast.alias):
        # Conservatively include the full import name and its possible top-level binding.
        return [node.asname, node.name, node.name.split(".")[0]]
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return [node.name]
    if isinstance(node, ast.MatchMapping):
        return [node.rest]
    return []


class ParameterDictionaries:
    """Keep original AST values; never evaluate expressions or mutate the source tree.

    A dictionary must have one simple assignment and exactly one name load in the
    entire scope subtree. Even later mutations/escapes reject a candidate. This is
    intentionally stricter than Python data flow, including nested name shadowing.
    """

    def __init__(self, tree):
        self.resolved = {}
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nodes = list(ast.walk(scope))
            references = defaultdict(list)
            blocked = set()
            for node in nodes:
                if isinstance(node, ast.Name):
                    references[node.id].append(node)
                blocked.update(other_bindings(node))
            # Reflective namespace access or star imports defeat this local assumption.
            if "*" in blocked or any(
                name in references for name in ("exec", "eval", "globals", "locals", "vars")
            ):
                continue
            positions = {statement: index for index, statement in enumerate(scope.body)}
            for index, statement in enumerate(scope.body):
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                elif isinstance(statement, ast.AnnAssign):
                    target = statement.target
                else:
                    continue
                if not isinstance(target, ast.Name) or not isinstance(statement.value, ast.Dict):
                    continue
                uses = references[target.id]
                loads = [use for use in uses if isinstance(use.ctx, ast.Load)]
                if target.id in blocked or len(uses) != 2 or len(loads) != 1:
                    continue
                use = loads[0]
                parent = parents[use]
                if not (isinstance(parent, ast.Call) and use in parent.args) and not (
                    isinstance(parent, ast.keyword) and parent.arg in (None, "params")
                ):
                    continue
                cursor = use
                while parents.get(cursor) is not scope:
                    cursor = parents.get(cursor)
                    if cursor is None or isinstance(
                        cursor,
                        (
                            ast.Lambda,
                            ast.ListComp,
                            ast.SetComp,
                            ast.DictComp,
                            ast.GeneratorExp,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                        ),
                    ):
                        break
                else:
                    # No branches, loops, with/try blocks or deferred function bodies.
                    if (
                        isinstance(cursor, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return))
                        and positions.get(cursor, -1) > index
                    ):
                        self.resolved[use] = statement.value

    def resolve(self, node):
        return self.resolved.get(node, node)
