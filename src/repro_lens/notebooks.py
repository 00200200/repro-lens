"""Read Jupyter notebook code cells as one Python module without running them."""

from __future__ import annotations

import ast
import json
import re

# Cell magics whose body is still Python; any other %% cell (bash, html, ...) is skipped.
PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun"}
SHELL_ASSIGNMENT = re.compile(r"(\s*)([\w.\[\], ]+?)\s*=\s*[!%]")
HELP = re.compile(r"\s*(\?\??[\w.]+|[\w.]+\?\??)\s*")


def ipython_placeholders(lines: list[str]) -> list[str]:
    """Replace IPython-only lines, and their continuations, keeping indentation and numbering.

    As in IPython 7–9, a line magic or shell command ends at its line unless the line ends
    with a backslash. Brackets and quotes in the command never continue it, so the next
    line stays Python.
    """
    result, continued, indent = [], False, ""
    for line in lines:
        if continued:
            result.append(f"{indent}pass")
        elif line.strip().startswith(("%", "!")) or HELP.fullmatch(line):
            indent = line[: len(line) - len(line.lstrip())]
            result.append(f"{indent}pass")
        elif assignment := SHELL_ASSIGNMENT.match(line):
            indent = assignment[1]
            result.append(f"{indent}{assignment[2]} = None")
        else:
            result.append(line)
            continue
        continued = line.endswith("\\")
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
        source = cell.get("source", "")
        code = cell_lines(source)
        try:
            ast.parse("\n".join(code))
        except SyntaxError as exc:
            original = "".join(source) if isinstance(source, list) else source
            skipped = code != (original.splitlines() or [""])
            context = " after skipping IPython commands" if skipped else ""
            invalid.append((number, exc.lineno or 1, f"{exc.msg}{context}"))
            code = ["pass"] * len(code)
        for line_number, line in enumerate(code, start=1):
            lines.append(line)
            locations.append((number, line_number))
    return "\n".join(lines) + "\n", locations, invalid
