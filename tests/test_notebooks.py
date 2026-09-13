import json

import pytest

from repro_lens.cli import main
from repro_lens.notebooks import notebook_source
from repro_lens.project import check, render


def notebook(*cells, cell_type="code"):
    return json.dumps(
        {
            "cells": [
                cell if isinstance(cell, dict) else {"cell_type": cell_type, "source": cell}
                for cell in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def markdown(text):
    return {"cell_type": "markdown", "source": text}


def test_findings_point_to_the_cell_and_line(tmp_path):
    (tmp_path / "explore.ipynb").write_text(
        notebook(
            ["from sklearn.model_selection import train_test_split\n", "import numpy as np\n"],
            markdown("train_test_split(X, y) in prose is not code"),
            "X, y = load()\ntrain, test = train_test_split(X, y)\n",
        )
    )
    report = check(tmp_path)
    assert report["files_checked"] == 1
    [finding] = report["findings"]
    assert (finding["path"], finding["cell"], finding["line"], finding["code"]) == (
        "explore.ipynb",
        3,
        2,
        "R101",
    )
    assert "explore.ipynb:cell 3:2:15 R101" in render(report, "text")


@pytest.mark.parametrize(
    "cell",
    [
        "%matplotlib inline",
        "!pip install scikit-learn",
        "files = !ls data",
        "timing = %timeit -o sum(range(10))",
        "for name in names:\n    !echo $name",
        "!saved_model_cli show --dir '{path}' \\\n                      --tag_set serve",
        "np.random?",
        "??train_test_split",
        "%%bash\nfor f in *.csv; do echo $f; done",
        "%%html\n<b>not python</b>",
    ],
)
def test_ipython_syntax_is_skipped_without_hiding_later_cells(tmp_path, cell):
    (tmp_path / "magic.ipynb").write_text(
        notebook(
            "import numpy as np\nnames = []",
            cell,
            "rng = np.random.default_rng()",
        )
    )
    assert [(f["code"], f["cell"]) for f in check(tmp_path)["findings"]] == [("R102", 3)]


@pytest.mark.parametrize(
    "cell, line",
    [
        ('!echo "("\nrandom.Random()', 2),
        ("!echo (\nrandom.Random()", 2),
        ("!echo hi  # (\nrandom.Random()", 2),
        ('%timeit f("(")\nrandom.Random()', 2),
        ("%time x = 1  # (\nrandom.Random()", 2),
        ('files = !ls "("\nrandom.Random()', 2),
        ('for name in names:\n    !echo "("\nrandom.Random()', 3),
        ("!pip install \\\n    numpy\nrandom.Random()", 3),
        ("!pip install \\\nnumpy \\\nscipy\nrandom.Random()", 4),
        ("%timeit f() \\\n  + 1\nrandom.Random()", 3),
        ("!pip install \\  \nrandom.Random()", 2),
    ],
)
def test_python_after_an_ipython_command_in_the_same_cell_is_checked(tmp_path, cell, line):
    # IPython 7.34, 8.12 and 9.17 end a line magic or shell command at its line unless
    # it ends with a backslash; quoted, commented or unbalanced brackets do not continue it.
    (tmp_path / "same_cell.ipynb").write_text(notebook("import random\nnames = []", cell))
    findings = check(tmp_path)["findings"]
    assert [(f["code"], f["cell"], f["line"]) for f in findings] == [("R103", 2, line)]


def test_bracket_continued_magic_is_reported_instead_of_hiding_code(tmp_path):
    # IPython keeps the second line as Python, so this cell is invalid when run.
    (tmp_path / "magic.ipynb").write_text(
        notebook(
            "import random",
            "%timeit sum((x for x in range(3)),\n    start=0)\nrandom.Random()",
            "random.Random()",
        )
    )
    findings = check(tmp_path)["findings"]
    assert [(f["code"], f["cell"]) for f in findings] == [("S902", 2), ("R103", 3)]
    assert "after skipping IPython commands" in findings[0]["message"]


def test_python_cell_magic_body_is_checked():
    source, locations, invalid = notebook_source(
        notebook("import random", "%%time\nrandom.Random()\n")
    )
    assert source.splitlines() == ["import random", "pass", "random.Random()"]
    assert locations == [(1, 1), (2, 1), (2, 2)]
    assert invalid == []


def test_valid_python_starting_with_percent_is_not_treated_as_a_magic():
    source, _, _ = notebook_source(notebook('label = ("%s"\n         % name)'))
    assert source.splitlines() == ['label = ("%s"', "         % name)"]


def test_invalid_cell_is_reported_while_other_cells_are_checked(tmp_path):
    (tmp_path / "lesson.ipynb").write_text(
        notebook(
            "import random",
            "data.loc[(:, 1), (:, 'HR')]",
            "random.Random()",
        )
    )
    findings = check(tmp_path)["findings"]
    assert [(f["code"], f["cell"], f["line"]) for f in findings] == [
        ("S902", 2, 1),
        ("R103", 3, 1),
    ]
    assert "was not checked" in findings[0]["message"]


def test_suppression_and_shadowing_work_across_cells(tmp_path):
    (tmp_path / "cells.ipynb").write_text(
        notebook(
            "import random",
            "random.Random()  # repro-lens: ignore[R103] -- Seeded by the course harness.",
            "random = make_rng()\nrandom.Random()",
        )
    )
    report = check(tmp_path)
    assert report["findings"] == []
    assert [(f["code"], f["cell"], f["line"]) for f in report["suppressed"]] == [("R103", 2, 1)]


def test_python_files_keep_their_report_shape(tmp_path):
    (tmp_path / "train.py").write_text("import random\nrandom.Random()")
    (tmp_path / "explore.ipynb").write_text(notebook("import random\nrandom.Random()"))
    findings = check(tmp_path)["findings"]
    assert [f["path"] for f in findings] == ["explore.ipynb", "train.py"]
    assert "cell" in findings[0] and "cell" not in findings[1]


def test_checkpoints_are_skipped_and_selected_notebooks_are_checked(tmp_path, capsys):
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / ".ipynb_checkpoints/explore-checkpoint.ipynb").write_text(
        notebook("import random\nrandom.Random()")
    )
    (tmp_path / "explore.ipynb").write_text(notebook("import random\nrandom.Random(1)"))
    assert check(tmp_path)["files_checked"] == 1
    assert main(["check", "explore.ipynb", "--root", str(tmp_path)]) == 0
    assert "1 Python files and notebooks checked" in capsys.readouterr().out


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "Expecting property name"),
        ('{"worksheets": []}', "nbformat 4"),
        (notebook("def broken(:\n    pass"), "not valid Python"),
        ('{"cells": [{"cell_type": "code", "source": 3}]}', "must be text"),
    ],
)
def test_unreadable_notebooks_are_not_a_clean_scan(tmp_path, content, message):
    (tmp_path / "broken.ipynb").write_text(content)
    [finding] = check(tmp_path)["findings"]
    assert finding["code"] == "S902"
    assert message in finding["message"]
    assert main(["check", "--root", str(tmp_path), "--fail-on", "error"]) == 1
