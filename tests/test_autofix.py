import ast
from pathlib import Path

from repro_lens.autofix import (
    LosslessCST,
    find_call_tokens,
    lossless_apply_call_fix,
    lossless_insert_call_argument,
    lossless_replace_call_argument,
)
from repro_lens.project import suggest_project_fixes


def _get_call(source: str, index: int = 0) -> ast.Call:
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    return calls[index]


def test_lossless_insert_single_line_empty_call():
    source = "f()"
    call = _get_call(source)
    res = lossless_insert_call_argument(source, call, "seed=42")
    assert res == "f(seed=42)"
    ast.parse(res)


def test_lossless_insert_single_line_with_args():
    source = "train_test_split(X, y)"
    call = _get_call(source)
    res = lossless_insert_call_argument(source, call, "random_state=42")
    assert res == "train_test_split(X, y, random_state=42)"
    ast.parse(res)


def test_lossless_insert_single_line_with_trailing_comma():
    source = "train_test_split(X, y, )"
    call = _get_call(source)
    res = lossless_insert_call_argument(source, call, "random_state=42")
    assert res == "train_test_split(X, y, random_state=42)"
    ast.parse(res)


def test_lossless_insert_multiline_with_trailing_comma_and_comment():
    source = (
        "train_test_split(\n"
        "    X,\n"
        "    y,  # target labels\n"
        ")\n"
    )
    call = _get_call(source)
    res = lossless_insert_call_argument(source, call, "random_state=42")
    expected = (
        "train_test_split(\n"
        "    X,\n"
        "    y,  # target labels\n"
        "    random_state=42,\n"
        ")\n"
    )
    assert res == expected
    ast.parse(res)


def test_lossless_insert_multiline_without_trailing_comma_with_comment():
    source = (
        "train_test_split(\n"
        "    X,\n"
        "    y  # target labels without comma\n"
        ")\n"
    )
    call = _get_call(source)
    res = lossless_insert_call_argument(source, call, "random_state=42")
    expected = (
        "train_test_split(\n"
        "    X,\n"
        "    y,  # target labels without comma\n"
        "    random_state=42,\n"
        ")\n"
    )
    assert res == expected
    ast.parse(res)


def test_lossless_replace_single_line_none():
    source = "train_test_split(X, y, random_state=None)"
    call = _get_call(source)
    res = lossless_replace_call_argument(source, call, "random_state", "42")
    assert res == "train_test_split(X, y, random_state=42)"
    ast.parse(res)


def test_lossless_replace_multiline_with_comment():
    source = (
        "train_test_split(\n"
        "    X,\n"
        "    y,\n"
        "    random_state=None,  # unseeded baseline\n"
        ")\n"
    )
    call = _get_call(source)
    res = lossless_replace_call_argument(source, call, "random_state", "42")
    expected = (
        "train_test_split(\n"
        "    X,\n"
        "    y,\n"
        "    random_state=42,  # unseeded baseline\n"
        ")\n"
    )
    assert res == expected
    ast.parse(res)


def test_lossless_replace_expression_value():
    source = "train_test_split(X, y, random_state=get_unseeded())"
    call = _get_call(source)
    res = lossless_replace_call_argument(source, call, "random_state", "42")
    assert res == "train_test_split(X, y, random_state=42)"
    ast.parse(res)


def test_lossless_apply_call_fix_inserts_or_replaces():
    source_new = "train_test_split(X, y)"
    call_new = _get_call(source_new)
    res_new = lossless_apply_call_fix(source_new, call_new, "random_state=42")
    assert res_new == "train_test_split(X, y, random_state=42)"

    source_existing = "train_test_split(X, y, random_state=None)"
    call_existing = _get_call(source_existing)
    res_existing = lossless_apply_call_fix(source_existing, call_existing, "random_state=42")
    assert res_existing == "train_test_split(X, y, random_state=42)"


def test_lossless_cst_class_find_and_diff():
    source = "def train():\n    train_test_split(X, y)\n"
    cst = LosslessCST(source)
    call = cst.find_call(2, 5)
    assert call is not None
    updated = cst.apply_call_fix(call, "random_state=42")
    assert "train_test_split(X, y, random_state=42)" in updated
    diff = cst.render_diff("train.py")
    assert "--- a/train.py" in diff
    assert "+    train_test_split(X, y, random_state=42)" in diff


def test_nested_calls_correct_target():
    source = "DataLoader(TensorDataset(X, y), shuffle=True)"
    tree = ast.parse(source)
    dataset_call = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "TensorDataset"
    ][0]
    res = lossless_insert_call_argument(source, dataset_call, "seed=42")
    assert res == "DataLoader(TensorDataset(X, y, seed=42), shuffle=True)"
    ast.parse(res)


def test_suggest_project_fixes_preserves_comments(tmp_path: Path):
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "train_test_split(\n"
        "    X,\n"
        "    y,  # target column\n"
        ")\n"
    )
    (tmp_path / "model.py").write_text(source, encoding="utf-8")
    modifications, report = suggest_project_fixes(tmp_path, seed=42)
    assert "model.py" in modifications
    fixed = modifications["model.py"]
    assert "# target column" in fixed
    assert "random_state=42," in fixed
    ast.parse(fixed)
