import json
from pathlib import Path

from repro_lens.cli import ansi_enabled, main, resolve_check_format
from repro_lens.project import check, render


def test_scan_never_executes_target_code(tmp_path):
    marker = tmp_path / "executed"
    source = (
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "import numpy as np\nnp.random.default_rng()"
    )
    (tmp_path / "train.py").write_text(source)
    report = check(tmp_path)
    assert [f["code"] for f in report["findings"]] == ["R102"]
    assert not marker.exists()


def test_policy_required_files_and_exclusions(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.repro-lens]\nrequired-files=["docs/reproduce.md"]\nexclude=["vendor/**"]\n'
    )
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor/bad.py").write_text("import random\nrandom.Random()")
    (tmp_path / "good.py").write_text("import random\nrandom.Random(1)")
    report = check(tmp_path)
    assert report["files_checked"] == 1
    assert [f["code"] for f in report["findings"]] == ["P201"]


def test_no_policy_does_not_require_a_particular_layout(tmp_path):
    (tmp_path / "one_file.py").write_text("print('simple project')")
    assert check(tmp_path)["findings"] == []


def test_invalid_toml_and_unknown_policy_are_visible(tmp_path):
    for content in ("[broken", "[tool.repro-lens]\nrequired_file=[]", 'tool="bad"'):
        (tmp_path / "pyproject.toml").write_text(content)
        assert check(tmp_path)["findings"][0]["code"] == "P202"


def test_explicit_paths_and_cli_json(tmp_path, capsys):
    (tmp_path / "bad.py").write_text("import random\nrandom.Random()")
    (tmp_path / "good.py").write_text("import random\nrandom.Random(2)")
    assert main(["check", "good.py", "--root", str(tmp_path), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["files_checked"] == 1
    assert main(["check", "--root", str(tmp_path)]) == 1
    assert main(["check", "--root", str(tmp_path), "--fail-on", "error"]) == 0


def test_review_is_not_a_blocking_error(tmp_path):
    (tmp_path / "train.py").write_text("from sklearn.model_selection import KFold\nKFold(**config)")
    assert main(["check", "--root", str(tmp_path)]) == 0


def test_init_will_not_overwrite_existing_project(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep me")
    assert main(["init", str(tmp_path)]) == 2
    assert marker.read_text() == "keep me"


def test_project_inside_ignored_parent_is_still_scanned(tmp_path):
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    (tmp_path / ".gitignore").write_text("scratch/\n")
    child = tmp_path / "scratch"
    child.mkdir()
    (child / "train.py").write_text("import random\nrandom.Random()")
    report = check(child)
    assert report["files_checked"] == 1
    assert report["findings"][0]["code"] == "R103"


def test_invalid_template_name_is_rejected_before_writing(tmp_path):
    for name in ("class", "../escape", "with-dash"):
        destination = tmp_path / name.replace("/", "_")
        assert main(["init", str(destination), "--name", name]) == 2
        assert not destination.exists()


def test_skill_launcher_uses_same_engine(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "train.py").write_text("import random\nrandom.Random()")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "skills/reproducibility/scripts/run.py"),
            "check",
            "--root",
            str(tmp_path),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["findings"] == check(tmp_path)["findings"]


def test_add_ignores_appends_suppression_comments(tmp_path):
    source = "import random\nrandom.Random()\nimport numpy as np\nnp.random.default_rng()\n"
    (tmp_path / "train.py").write_text(source)
    assert main(["check", "--root", str(tmp_path), "--add-ignores"]) == 0
    updated = (tmp_path / "train.py").read_text()
    assert (
        "random.Random()  # repro-lens: ignore[R103] -- TODO: Review reproducibility\n" in updated
    )
    assert (
        "np.random.default_rng()  # repro-lens: ignore[R102] -- TODO: Review reproducibility\n"
        in updated
    )
    report = check(tmp_path)
    assert report["findings"] == []
    assert {f["code"] for f in report["suppressed"]} == {"R102", "R103"}


def test_add_ignores_merges_codes_on_one_line(tmp_path):
    (tmp_path / "train.py").write_text(
        "import random\nimport numpy as np\nrandom.Random(); np.random.default_rng()\n"
    )
    assert main(["check", "--root", str(tmp_path), "--add-ignores"]) == 0
    line = (tmp_path / "train.py").read_text().splitlines()[2]
    assert line.endswith("# repro-lens: ignore[R102, R103] -- TODO: Review reproducibility")
    report = check(tmp_path)
    assert report["findings"] == []
    assert sorted(f["code"] for f in report["suppressed"]) == ["R102", "R103"]


def test_add_ignores_skips_notebooks_and_policy_findings(tmp_path):
    import json as json_mod

    (tmp_path / "pyproject.toml").write_text(
        '[tool.repro-lens]\nrequired-files=["docs/reproduce.md"]\n'
    )
    (tmp_path / "train.py").write_text("import random\nrandom.Random()\n")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["import random\n", "random.Random()\n"],
                "outputs": [],
                "execution_count": None,
            }
        ],
    }
    (tmp_path / "demo.ipynb").write_text(json_mod.dumps(notebook))
    before_nb = (tmp_path / "demo.ipynb").read_text()
    assert main(["check", "--root", str(tmp_path), "--add-ignores"]) == 1
    assert "repro-lens: ignore[R103]" in (tmp_path / "train.py").read_text()
    assert (tmp_path / "demo.ipynb").read_text() == before_nb
    assert [f["code"] for f in check(tmp_path)["findings"] if f["code"] == "P201"] == ["P201"]


def test_add_ignores_is_idempotent(tmp_path):
    path = tmp_path / "train.py"
    path.write_text("import random\nrandom.Random()\n")
    assert main(["check", "--root", str(tmp_path), "--add-ignores"]) == 0
    once = path.read_text()
    assert main(["check", "--root", str(tmp_path), "--add-ignores"]) == 0
    assert path.read_text() == once
    assert once.count("repro-lens: ignore") == 1


def test_pretty_underlines_the_unseeded_call(tmp_path):
    (tmp_path / "train.py").write_text("import random\nx = random.Random()\n")
    output = render(check(tmp_path), "pretty")
    assert "warning[R103]: random.Random has no explicit non-None seed." in output
    assert "--> train.py:2:5" in output
    assert "2 | x = random.Random()" in output
    assert "  |     ^^^^^^^^^^^^^^^" in output
    assert "  = help: Pass the experiment's seed/RNG" in output
    assert "\033" not in output


def test_pretty_color_marks_only_the_label_and_carets(tmp_path):
    (tmp_path / "train.py").write_text("import random\nrandom.Random()\n")
    output = render(check(tmp_path), "pretty", color=True)
    assert "\033[1;33mwarning[R103]\033[0m" in output
    assert "2 | random.Random()" in output
    assert "\033[1;33m^^^^^^^^^^^^^^^\033[0m" in output


def test_pretty_notebook_uses_the_cell_line(tmp_path):
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# notes"]},
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["import random\n", "random.Random()\n"],
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    (tmp_path / "notes.ipynb").write_text(json.dumps(notebook))
    output = render(check(tmp_path), "pretty")
    assert "--> notes.ipynb:cell 2:2:1" in output
    assert "2 | random.Random()" in output


def test_pretty_omits_a_snippet_when_the_file_is_missing(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.repro-lens]\nrequired-files=["docs/reproduce.md"]\n'
    )
    output = render(check(tmp_path), "pretty")
    assert "error[P201]:" in output
    assert "--> docs/reproduce.md:1:1" in output
    assert " = help: Create it or deliberately revise the project policy." in output
    assert "|" not in output.split("error[P201]", 1)[1]


def test_check_format_defaults(monkeypatch):
    assert resolve_check_format(None, writing_file=False, is_tty=True) == "pretty"
    assert resolve_check_format(None, writing_file=False, is_tty=False) == "text"
    assert resolve_check_format(None, writing_file=True, is_tty=True) == "text"
    assert resolve_check_format("text", writing_file=False, is_tty=True) == "text"
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert ansi_enabled(writing_file=False, is_tty=True) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert ansi_enabled(writing_file=False, is_tty=True) is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "0")
    assert ansi_enabled(writing_file=False, is_tty=True) is False
    assert ansi_enabled(writing_file=True, is_tty=True) is False


def test_pretty_cli_stays_plain_when_stdout_is_captured(tmp_path, capsys):
    (tmp_path / "train.py").write_text("import random\nrandom.Random()\n")
    assert main(["check", "--root", str(tmp_path), "--format", "pretty"]) == 1
    pretty = capsys.readouterr().out
    assert "warning[R103]" in pretty
    assert "\033" not in pretty
    assert main(["check", "--root", str(tmp_path)]) == 1
    text = capsys.readouterr().out
    assert "train.py:2:1 R103 [warning]" in text
    assert "-->" not in text
