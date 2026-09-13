import json
from pathlib import Path

from repro_lens.cli import main
from repro_lens.project import check


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
