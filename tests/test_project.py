import json
from pathlib import Path

import pytest

from repro_lens.cli import ansi_enabled, main, resolve_check_format
from repro_lens.project import check, read_policy, render


def test_parallel_scan_matches_serial_findings_and_order(tmp_path):
    for index in range(25):
        (tmp_path / f"file_{index:02d}.py").write_text(
            "from sklearn.model_selection import train_test_split\n"
            "train_test_split(data, labels)\n",
            encoding="utf-8",
        )
    serial = check(tmp_path, jobs=1)
    parallel = check(tmp_path, jobs=2)
    assert parallel["files_checked"] == serial["files_checked"] == 25
    assert parallel["findings"] == serial["findings"]
    assert parallel["suppressed"] == serial["suppressed"]


def test_parallel_scan_rejects_non_positive_jobs(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        check(tmp_path, jobs=0)


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


def test_policy_ignore_rules(tmp_path):
    (tmp_path / "train.py").write_text("import random\nrandom.Random()\n")
    # Without policy, R103 is flagged
    assert [f["code"] for f in check(tmp_path)["findings"]] == ["R103"]

    # With ignore = ["R103"], R103 is skipped
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens]\nignore = ["R103"]\n')
    policy = read_policy(tmp_path)
    assert policy["ignore"] == ["R103"]
    assert check(tmp_path)["findings"] == []

    # Invalid rule code in ignore raises ValueError in read_policy and yields P202 in check()
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens]\nignore = ["R999"]\n')
    with pytest.raises(ValueError, match="Unknown rule code in ignore: R999"):
        read_policy(tmp_path)
    report = check(tmp_path)
    assert any(f["code"] == "P202" for f in report["findings"])

    # Non-list ignore raises ValueError
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens]\nignore = "R103"\n')
    with pytest.raises(ValueError, match="ignore must be a list"):
        read_policy(tmp_path)


def test_policy_severity_overrides(tmp_path):
    (tmp_path / "train.py").write_text("import random\nrandom.Random()\n")

    # Override R103 (default: warning) to error
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens.severities]\nR103 = "error"\n')
    policy = read_policy(tmp_path)
    assert policy["severities"] == {"R103": "error"}
    report = check(tmp_path)
    assert len(report["findings"]) == 1
    assert report["findings"][0]["code"] == "R103"
    assert report["findings"][0]["severity"] == "error"
    # Fails when failing on error
    assert main(["check", "--root", str(tmp_path), "--fail-on", "error"]) == 1

    # Override R103 to review
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens.severities]\nR103 = "review"\n')
    report = check(tmp_path)
    assert report["findings"][0]["severity"] == "review"
    # Does not fail when fail-on is warning
    assert main(["check", "--root", str(tmp_path), "--fail-on", "warning"]) == 0

    # Invalid severity string raises ValueError
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens.severities]\nR103 = "fatal"\n')
    with pytest.raises(ValueError, match="Invalid severity for R103"):
        read_policy(tmp_path)

    # Unknown rule in severities raises ValueError
    (tmp_path / "pyproject.toml").write_text('[tool.repro-lens.severities]\nR999 = "warning"\n')
    with pytest.raises(ValueError, match="Unknown rule code in severities: R999"):
        read_policy(tmp_path)


def test_check_diff_prints_unified_diff_and_leaves_files_untouched(tmp_path, capsys):
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "clf = RandomForestClassifier()\n"
    )
    file_path = tmp_path / "train.py"
    file_path.write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--diff"])
    captured = capsys.readouterr()
    assert code == 1
    assert "--- a/train.py" in captured.out
    assert "+++ b/train.py" in captured.out
    assert (
        "+X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)"
        in captured.out
    )
    assert "+clf = RandomForestClassifier(random_state=42)" in captured.out
    assert file_path.read_text(encoding="utf-8") == source


def test_check_suggest_alias_matches_diff(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")

    code_diff = main(["check", "--root", str(tmp_path), "--diff"])
    out_diff = capsys.readouterr().out

    code_suggest = main(["check", "--root", str(tmp_path), "--suggest"])
    out_suggest = capsys.readouterr().out

    assert code_diff == code_suggest == 1
    assert out_diff == out_suggest
    assert "train_test_split(X, y, random_state=42)" in out_diff


def test_check_diff_preserves_multiline_indentation_and_existing_arguments(tmp_path, capsys):
    source = (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "clf = RandomForestClassifier(\n"
        "    n_estimators=100,\n"
        "    max_depth=5,\n"
        ")\n\n"
        "split = train_test_split(\n"
        "    features,\n"
        "    labels\n"
        ")\n\n"
        "clf2 = RandomForestClassifier(\n"
        "    n_estimators=50)\n"
    )
    (tmp_path / "model.py").write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--diff"])
    captured = capsys.readouterr()
    assert code == 1
    assert "    max_depth=5,\n+    random_state=42,\n" in captured.out
    assert "    labels,\n+    random_state=42,\n" in captured.out
    assert "    n_estimators=50, random_state=42)" in captured.out


def test_check_diff_custom_seed(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--diff", "--seed", "1337"])
    captured = capsys.readouterr()
    assert code == 1
    assert "train_test_split(X, y, random_state=1337)" in captured.out


def test_check_diff_replaces_existing_none_seed(tmp_path, capsys):
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "train_test_split(X, y, random_state=None)\n"
    )
    (tmp_path / "train.py").write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--diff"])
    captured = capsys.readouterr()
    assert code == 1
    assert (
        "-train_test_split(X, y, random_state=None)\n+train_test_split(X, y, random_state=42)"
        in captured.out
    )


def test_check_fix_modifies_files_in_place_and_resolves_findings(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    file_path = tmp_path / "train.py"
    file_path.write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--fix"])
    assert code == 0
    updated = file_path.read_text(encoding="utf-8")
    assert "train_test_split(X, y, random_state=42)" in updated
    assert check(tmp_path)["findings"] == []


def test_check_diff_and_fix_together(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    file_path = tmp_path / "train.py"
    file_path.write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--diff", "--fix"])
    captured = capsys.readouterr()
    assert code == 0
    assert "--- a/train.py" in captured.out
    assert "+train_test_split(X, y, random_state=42)" in captured.out
    assert "train_test_split(X, y, random_state=42)" in file_path.read_text(encoding="utf-8")
    assert check(tmp_path)["findings"] == []


def test_check_diff_output_file(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")
    patch_file = tmp_path / "suggested.diff"

    code = main(["check", "--root", str(tmp_path), "--diff", "--output", str(patch_file)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert patch_file.is_file()
    patch_content = patch_file.read_text(encoding="utf-8")
    assert "--- a/train.py" in patch_content
    assert "+train_test_split(X, y, random_state=42)" in patch_content


def test_check_diff_clean_project_returns_zero_and_empty_diff(tmp_path, capsys):
    source = "print('reproducible code without stochastic calls')\n"
    (tmp_path / "clean.py").write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--diff"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_check_diff_skips_notebooks(tmp_path, capsys):
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from sklearn.model_selection import train_test_split\n",
                    "train_test_split(X, y)\n",
                ],
                "outputs": [],
                "execution_count": 1,
            }
        ],
    }
    nb_path = tmp_path / "experiment.ipynb"
    nb_path.write_text(json.dumps(notebook), encoding="utf-8")
    nb_before = nb_path.read_text(encoding="utf-8")

    main(["check", "--root", str(tmp_path), "--diff"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert nb_path.read_text(encoding="utf-8") == nb_before


def test_write_baseline_generates_portable_json(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--write-baseline"])
    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 1 baseline findings to .repro-lens-baseline.json" in captured.out

    baseline_file = tmp_path / ".repro-lens-baseline.json"
    assert baseline_file.is_file()
    data = json.loads(baseline_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["generator"] == "repro-lens"
    assert len(data["findings"]) == 1
    assert data["findings"][0]["path"] == "train.py"
    assert data["findings"][0]["code"] == "R101"


def test_write_baseline_custom_path(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")

    code = main(["check", "--root", str(tmp_path), "--write-baseline", "custom_baseline.json"])
    assert code == 0
    assert (tmp_path / "custom_baseline.json").is_file()


def test_baseline_suppresses_legacy_findings(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")

    # Record baseline
    main(["check", "--root", str(tmp_path), "--write-baseline"])

    # Normal check without baseline flag fails
    assert main(["check", "--root", str(tmp_path)]) == 1

    # Check with baseline flag passes
    assert main(["check", "--root", str(tmp_path), "--baseline"]) == 0
    report = check(tmp_path, baseline=tmp_path / ".repro-lens-baseline.json")
    assert report["findings"] == []
    assert len(report["suppressed"]) == 1


def test_baseline_reports_new_regressions(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")
    main(["check", "--root", str(tmp_path), "--write-baseline"])

    # Add a new regression
    (tmp_path / "model.py").write_text(
        "from sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier()\n",
        encoding="utf-8",
    )

    code = main(["check", "--root", str(tmp_path), "--baseline"])
    captured = capsys.readouterr()
    assert code == 1
    assert "model.py" in captured.out
    assert "train.py" not in captured.out


def test_baseline_in_pyproject_toml_is_automatically_loaded(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")
    main(["check", "--root", str(tmp_path), "--write-baseline"])

    (tmp_path / "pyproject.toml").write_text(
        '[tool.repro-lens]\nbaseline = ".repro-lens-baseline.json"\n',
        encoding="utf-8",
    )

    # Automatically passes without explicit --baseline
    assert main(["check", "--root", str(tmp_path)]) == 0


def test_baseline_shifted_lines_still_suppressed(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")
    main(["check", "--root", str(tmp_path), "--write-baseline"])

    # Insert lines above
    (tmp_path / "train.py").write_text(
        "# New comment 1\n# New comment 2\n" + source,
        encoding="utf-8",
    )

    assert main(["check", "--root", str(tmp_path), "--baseline"]) == 0


def test_clean_baseline_removes_stale_entries(tmp_path, capsys):
    source = "from sklearn.model_selection import train_test_split\ntrain_test_split(X, y)\n"
    (tmp_path / "train.py").write_text(source, encoding="utf-8")
    main(["check", "--root", str(tmp_path), "--write-baseline"])

    # Fix the issue
    (tmp_path / "train.py").write_text(
        "from sklearn.model_selection import train_test_split\n"
        "train_test_split(X, y, random_state=42)\n",
        encoding="utf-8",
    )

    assert main(["check", "--root", str(tmp_path), "--baseline", "--clean-baseline"]) == 0
    cleaned = json.loads((tmp_path / ".repro-lens-baseline.json").read_text(encoding="utf-8"))
    assert cleaned["findings"] == []


def test_baseline_missing_file_raises_error(tmp_path, capsys):
    assert main(["check", "--root", str(tmp_path), "--baseline", "missing.json"]) == 2
    captured = capsys.readouterr()
    assert "Baseline file does not exist" in captured.err


def test_suggest_fixes_for_distributed_sampler_r121(tmp_path):
    from repro_lens.project import suggest_project_fixes

    source = (
        "from torch.utils.data.distributed import DistributedSampler\n"
        "sampler = DistributedSampler(dataset)\n"
    )
    (tmp_path / "train.py").write_text(source, encoding="utf-8")
    modifications, report = suggest_project_fixes(tmp_path, seed=42)
    assert "train.py" in modifications
    assert "DistributedSampler(dataset, seed=42)" in modifications["train.py"]
