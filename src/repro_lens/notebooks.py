"""Read Jupyter notebook code cells as one Python module without running them."""

from __future__ import annotations

import ast
import json
import re

# Cell magics whose body is still Python; any other %% cell (bash, html, ...) is skipped.
PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun"}
SHELL_ASSIGNMENT = re.compile(r"(\s*)([\w.\[\], ]+?)\s*=\s*[!%]")
HELP = re.compile(r"\s*(\?\??[\w.]+|[\w.]+\?\??)\s*")
OPEN, CLOSE = "([{", ")]}"


def ipython_placeholders(lines: list[str]) -> list[str]:
    """Replace IPython-only lines, and their continuations, keeping indentation and numbering."""
    result, depth, continued, indent = [], 0, False, ""
    for line in lines:
        stripped = line.strip()
        if continued:
            result.append(f"{indent}pass")
        elif stripped.startswith(("%", "!")) or HELP.fullmatch(line):
            indent = line[: len(line) - len(line.lstrip())]
            result.append(f"{indent}pass")
            depth = 0
        elif assignment := SHELL_ASSIGNMENT.match(line):
            indent = assignment[1]
            result.append(f"{indent}{assignment[2]} = None")
            depth = 0
        else:
            result.append(line)
            continue
        # A magic or shell command continues over open brackets or a trailing backslash.
        depth += sum(line.count(c) for c in OPEN) - sum(line.count(c) for c in CLOSE)
        continued = depth > 0 or line.rstrip().endswith("\\")
    return result


def cell_lines(source: object) -> list[str]:
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        source = "".join(source)
    if not isinstance(source, str):
        raise ValueError("Notebook cell source must be text")
    lines = source.splitlines() or [""]
    if lines[0].startswith("%%"):
        magic = lines[0][2:].split(maxsplit=1)[0] if lines[0][2:].strip() else ""
        if magic not in PYTHON_CELL_MAGICS:
            return ["pass"] * len(lines)
        lines[0] = "pass"
    try:
        # Valid Python is kept as is, so a continuation line starting with % stays intact.
        ast.parse("\n".join(lines))
        return lines
    except SyntaxError:
        return ipython_placeholders(lines)


def notebook_source(
    text: str,
) -> tuple[str, list[tuple[int, int]], list[tuple[int, int, str]]]:
    """Join code cells in notebook order.

    Returns the joined source, a (cell number, cell line) location for each joined line,
    and (cell number, cell line, message) for cells that are not valid Python. Those cells
    are left out so the remaining cells can still be checked.
    """
    document = json.loads(text)
    cells = document.get("cells") if isinstance(document, dict) else None
    if not isinstance(cells, list):
        raise ValueError("Only nbformat 4 notebooks with a cells list are supported")
    lines, locations, invalid = [], [], []
    for number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        code = cell_lines(cell.get("source", ""))
        try:
            ast.parse("\n".join(code))
        except SyntaxError as exc:
            invalid.append((number, exc.lineno or 1, exc.msg))
            code = ["pass"] * len(code)
        for line_number, line in enumerate(code, start=1):
            lines.append(line)
            locations.append((number, line_number))
    return "\n".join(lines) + "\n", locations, invalid
