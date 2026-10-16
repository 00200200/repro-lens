"""Lossless Concrete Syntax Tree (CST) and token-based autofix transformer.

Preserves exact comments, token whitespace, and indentation when injecting or
updating reproducibility arguments (e.g., random_state=42, seed=42) or
suppression comments across Python source files.
"""

from __future__ import annotations

import ast
import difflib
import io
import token
import tokenize
from typing import Sequence


def compute_line_offsets(source: str) -> list[int]:
    """Compute 0-based cumulative character offsets for line starts (1-indexed lines)."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets


def pos_to_offset(pos: tuple[int, int], offsets: list[int]) -> int:
    """Convert a 1-indexed (row, col) token position to a 0-based character offset."""
    row, col = pos
    if row <= 0:
        return col
    if row - 1 < len(offsets):
        return offsets[row - 1] + col
    return offsets[-1] + col


def find_call_tokens(
    source: str, call: ast.Call
) -> tuple[tokenize.TokenInfo | None, tokenize.TokenInfo | None, list[tokenize.TokenInfo]]:
    """Locate the opening parenthesis, closing parenthesis, and inner tokens for an ast.Call."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return None, None, []

    func_end = (
        getattr(call.func, "end_lineno", call.lineno),
        getattr(call.func, "end_col_offset", call.col_offset),
    )
    open_idx: int | None = None
    for i, tok in enumerate(tokens):
        if tok.start >= func_end and tok.string == "(":
            open_idx = i
            break

    if open_idx is None:
        return None, None, []

    depth = 0
    close_idx: int | None = None
    for i in range(open_idx, len(tokens)):
        tok = tokens[i]
        if tok.string in ("(", "[", "{"):
            depth += 1
        elif tok.string in (")", "]", "}"):
            depth -= 1
            if depth == 0:
                close_idx = i
                break

    if close_idx is None:
        return None, None, []

    return tokens[open_idx], tokens[close_idx], tokens[open_idx + 1 : close_idx]


def lossless_insert_call_argument(source: str, call: ast.Call, kwarg_str: str) -> str:
    """Safely insert a keyword argument into an ast.Call, preserving comments and formatting."""
    offsets = compute_line_offsets(source)
    open_tok, close_tok, inner = find_call_tokens(source, call)
    if open_tok is None or close_tok is None:
        return source

    substantive = [
        t
        for t in inner
        if t.type
        not in (
            tokenize.COMMENT,
            tokenize.NL,
            token.NEWLINE,
            token.INDENT,
            token.DEDENT,
        )
    ]

    lines = source.splitlines(keepends=True)
    call_line = lines[call.lineno - 1] if 1 <= call.lineno <= len(lines) else ""
    call_indent = call_line[: len(call_line) - len(call_line.lstrip())]

    # Case 1: Empty call (e.g. func())
    if not substantive:
        if open_tok.start[0] == close_tok.start[0]:
            close_off = pos_to_offset(close_tok.start, offsets)
            return source[:close_off] + kwarg_str + source[close_off:]
        else:
            arg_indent = call_indent + "    "
            close_line_idx = close_tok.start[0] - 1
            close_line = lines[close_line_idx] if 0 <= close_line_idx < len(lines) else ""
            close_col = close_tok.start[1]
            if close_line[:close_col].strip() == "":
                insert_pos = pos_to_offset((close_tok.start[0], 0), offsets)
                return source[:insert_pos] + f"{arg_indent}{kwarg_str},\n" + source[insert_pos:]
            else:
                close_off = pos_to_offset(close_tok.start, offsets)
                return source[:close_off] + kwarg_str + source[close_off:]

    # Case 2: Call has arguments.
    last_arg = call.keywords[-1] if call.keywords else (call.args[-1] if call.args else None)

    # Check for trailing comma after last argument at depth 0 inside inner tokens
    inner_depth = 0
    last_comma_idx: int | None = None
    last_arg_token_idx: int | None = None
    for i, tok in enumerate(inner):
        if tok.type not in (
            tokenize.COMMENT,
            tokenize.NL,
            token.NEWLINE,
            token.INDENT,
            token.DEDENT,
        ):
            if inner_depth == 0 and tok.string == ",":
                last_comma_idx = i
            else:
                last_arg_token_idx = i
        if tok.string in ("(", "[", "{"):
            inner_depth += 1
        elif tok.string in (")", "]", "}"):
            inner_depth -= 1

    has_trailing_comma = (
        last_comma_idx is not None
        and last_arg_token_idx is not None
        and last_comma_idx > last_arg_token_idx
    )

    # Single-line call:
    if open_tok.start[0] == close_tok.start[0]:
        close_off = pos_to_offset(close_tok.start, offsets)
        before_paren = source[:close_off]
        if before_paren.rstrip().endswith(","):
            prefix = " " if not before_paren.endswith(" ") else ""
            return before_paren + prefix + kwarg_str + source[close_off:]
        else:
            return before_paren + ", " + kwarg_str + source[close_off:]

    # Multi-line call:
    res = source
    if not has_trailing_comma and last_arg_token_idx is not None:
        last_arg_tok = inner[last_arg_token_idx]
        comma_off = pos_to_offset(last_arg_tok.end, offsets)
        res = res[:comma_off] + "," + res[comma_off:]
        # Recalculate offsets and closing paren position
        offsets = compute_line_offsets(res)
        new_open, new_close, _ = find_call_tokens(res, call)
        if new_close is not None:
            close_tok = new_close

    lines = res.splitlines(keepends=True)
    if last_arg and getattr(last_arg, "lineno", call.lineno) > call.lineno:
        last_arg_line = lines[last_arg.lineno - 1]
        arg_indent = last_arg_line[: len(last_arg_line) - len(last_arg_line.lstrip())]
    else:
        arg_indent = call_indent + "    "

    close_line_idx = close_tok.start[0] - 1
    close_line = lines[close_line_idx] if 0 <= close_line_idx < len(lines) else ""
    close_col = close_tok.start[1]
    if close_line[:close_col].strip() == "":
        insert_pos = pos_to_offset((close_tok.start[0], 0), offsets)
        return res[:insert_pos] + f"{arg_indent}{kwarg_str},\n" + res[insert_pos:]
    else:
        close_off = pos_to_offset(close_tok.start, offsets)
        before_paren = res[:close_off]
        if before_paren.rstrip().endswith(","):
            prefix = " " if not before_paren.endswith(" ") else ""
            return before_paren + prefix + kwarg_str + res[close_off:]
        else:
            return before_paren + ", " + kwarg_str + res[close_off:]


def lossless_replace_call_argument(
    source: str, call: ast.Call, kwarg_name: str, new_value: str
) -> str:
    """Losslessly replace the value of an existing keyword argument in an ast.Call."""
    offsets = compute_line_offsets(source)
    open_tok, close_tok, inner = find_call_tokens(source, call)
    if open_tok is None or close_tok is None:
        return source

    inner_depth = 0
    kw_idx: int | None = None
    for i, tok in enumerate(inner):
        if inner_depth == 0 and tok.type == token.NAME and tok.string == kwarg_name:
            j = i + 1
            while j < len(inner) and inner[j].type in (
                tokenize.COMMENT,
                tokenize.NL,
                token.NEWLINE,
            ):
                j += 1
            if j < len(inner) and inner[j].string == "=":
                kw_idx = j
                break
        if tok.string in ("(", "[", "{"):
            inner_depth += 1
        elif tok.string in (")", "]", "}"):
            inner_depth -= 1

    if kw_idx is None:
        return source

    val_start_idx = kw_idx + 1
    while val_start_idx < len(inner) and inner[val_start_idx].type in (
        tokenize.COMMENT,
        tokenize.NL,
        token.NEWLINE,
    ):
        val_start_idx += 1
    if val_start_idx >= len(inner):
        return source

    val_end_idx = val_start_idx
    inner_depth = 0
    for i in range(val_start_idx, len(inner)):
        tok = inner[i]
        if inner_depth == 0 and tok.string == ",":
            break
        if tok.string in ("(", "[", "{"):
            inner_depth += 1
        elif tok.string in (")", "]", "}"):
            inner_depth -= 1
        if tok.type not in (tokenize.COMMENT, tokenize.NL, token.NEWLINE):
            val_end_idx = i

    start_off = pos_to_offset(inner[val_start_idx].start, offsets)
    end_off = pos_to_offset(inner[val_end_idx].end, offsets)
    return source[:start_off] + new_value + source[end_off:]


def lossless_apply_call_fix(source: str, call: ast.Call, kwarg_str: str) -> str:
    """Apply an automated keyword fix, updating an existing keyword or inserting a new one."""
    if "=" not in kwarg_str:
        return source
    kw_name, kw_val = kwarg_str.split("=", 1)
    existing_kw = next((k for k in call.keywords if k.arg == kw_name), None)
    if existing_kw:
        return lossless_replace_call_argument(source, call, kw_name, kw_val)
    return lossless_insert_call_argument(source, call, kwarg_str)


class LosslessCST:
    """Concrete Syntax Tree and token manager for lossless code fixes."""

    def __init__(self, source: str) -> None:
        self.initial_source = source
        self.source = source
        self._tree = ast.parse(source)

    @property
    def tree(self) -> ast.AST:
        return self._tree

    def find_call(self, line: int, col: int) -> ast.Call | None:
        """Find the ast.Call node matching 1-indexed line and col."""
        target_col = col - 1
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Call) and node.lineno == line and node.col_offset == target_col:
                return node
        line_calls = [
            n for n in ast.walk(self._tree) if isinstance(n, ast.Call) and n.lineno == line
        ]
        if line_calls:
            line_calls.sort(key=lambda n: abs(n.col_offset - target_col))
            return line_calls[0]
        span_calls = [
            n
            for n in ast.walk(self._tree)
            if isinstance(n, ast.Call)
            and n.lineno <= line <= getattr(n, "end_lineno", n.lineno)
        ]
        if span_calls:
            span_calls.sort(
                key=lambda n: (
                    getattr(n, "end_lineno", n.lineno) - n.lineno,
                    abs(n.col_offset - target_col),
                )
            )
            return span_calls[0]
        return None

    def apply_call_fix(self, call: ast.Call, kwarg_str: str) -> str:
        """Apply a fix to a call site and update the internal source and AST."""
        new_source = lossless_apply_call_fix(self.source, call, kwarg_str)
        if new_source != self.source:
            self.source = new_source
            self._tree = ast.parse(new_source)
        return self.source

    def render_diff(self, filename: str = "source.py") -> str:
        """Render a unified diff against the initial source."""
        orig_lines = self.initial_source.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                orig_lines,
                self.source.splitlines(keepends=True),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
            )
        )
