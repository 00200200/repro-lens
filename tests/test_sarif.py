import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from repro_lens.analysis import RULES
from repro_lens.cli import main
from repro_lens.project import check
from repro_lens.sarif import escape, to_github, to_sarif

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "from sklearn.tree import DecisionTreeClassifier\nDecisionTreeClassifier()\n"
NOTEBOOK = json.dumps(
    {
        "cells": [{"cell_type": "code", "source": "import random\n\nrandom.shuffle(items)"}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
)


def project(tmp_path):
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src/train.py").write_text(SCRIPT)
    (tmp_path / "explore.ipynb").write_text(NOTEBOOK)
    return tmp_path


def test_sarif_results_keep_locations_levels_and_rule_links(tmp_path):
    report = check(project(tmp_path))
    run = to_sarif(report)["runs"][0]
    rules = run["tool"]["driver"]["rules"]
    assert [rule["id"] for rule in rules] == list(RULES)
    by_rule = {result["ruleId"]: result for result in run["results"]}
    assert set(by_rule) == {"R101", "R112"}

    script = by_rule["R101"]
    assert script["level"] == "warning"
    assert rules[script["ruleIndex"]]["id"] == "R101"
    assert script["locations"][0]["physicalLocation"] == {
        "artifactLocation": {"uri": "src/train.py"},
        "region": {"startLine": 2, "startColumn": 1},
    }
    assert rules[script["ruleIndex"]]["helpUri"].endswith("/docs/rules.md")

    notebook = by_rule["R112"]
    assert notebook["level"] == "note"
    assert notebook["message"]["text"].startswith("Cell 1, line 3: ")
    assert notebook["locations"][0]["physicalLocation"]["region"] == {"startLine": 1}
    assert rules[notebook["ruleIndex"]]["helpUri"].endswith("/docs/frameworks.md")


def test_github_annotations_escape_workflow_command_syntax():
    report = {
        "files_checked": 1,
        "assurance": "Static screening only.",
        "findings": [
            {
                "path": "a,b:c.py",
                "line": 3,
                "column": 5,
                "code": "S902",
                "severity": "error",
                "message": "100% broken\nsecond line",
                "suggestion": "Fix it.",
            }
        ],
    }
    first, summary = to_github(report, "sub").splitlines()
    assert first == (
        "::error file=sub/a%2Cb%3Ac.py,line=3,col=5,title=Repro Lens S902::"
        "100%25 broken%0Asecond line Fix it."
    )
    assert summary == "Repro Lens checked 1 Python files and notebooks: 1 finding. " + (
        "Static screening only."
    )
    assert escape("a:b") == "a:b"


def test_cli_prefixes_paths_when_root_is_inside_the_working_directory(
    tmp_path, monkeypatch, capsys
):
    project(tmp_path / "service")
    monkeypatch.chdir(tmp_path)
    assert main(["check", "--root", "service", "--format", "github"]) == 1
    output = capsys.readouterr().out
    assert "::warning file=service/src/train.py,line=2,col=1,title=Repro Lens R101::" in output
    assert "::notice file=service/explore.ipynb,title=Repro Lens R112::Cell 1" in output

    monkeypatch.chdir(tmp_path / "service")
    assert main(["check", "--format", "sarif", "--fail-on", "error"]) == 0
    uris = [
        result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for result in json.loads(capsys.readouterr().out)["runs"][0]["results"]
    ]
    assert sorted(uris) == ["explore.ipynb", "src/train.py"]


def action_steps():
    return yaml.safe_load((ROOT / "action.yml").read_text())["runs"]["steps"]


def test_action_inputs_match_the_cli():
    action = yaml.safe_load((ROOT / "action.yml").read_text())
    assert set(action["inputs"]) == {"root", "fail-on", "sarif-file"}
    assert action["inputs"]["fail-on"]["default"] == "warning"
    setup = [step for step in action_steps() if "uses" in step]
    assert len(setup) == 1 and setup[0]["if"] == "steps.python.outputs.command == ''"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
@pytest.mark.parametrize(("fail_on", "status"), [("warning", 1), ("error", 0)])
def test_action_scripts_annotate_write_sarif_and_fail_as_configured(tmp_path, fail_on, status):
    workspace = tmp_path / "workspace"
    project(workspace / "service")
    output = tmp_path / "github_output"
    output.write_text("")
    finder, _, checker = action_steps()
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    }
    found = subprocess.run(["bash", "-eo", "pipefail", "-c", finder["run"]], env=env, check=True)
    assert found.returncode == 0
    command = output.read_text().strip().removeprefix("command=")
    assert command

    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", checker["run"]],
        cwd=workspace,
        env={
            **env,
            "PYTHON": sys.executable,
            "ROOT": "service",
            "FAIL_ON": fail_on,
            "SARIF_FILE": "repro-lens.sarif",
            "PYTHONPATH": str(ROOT / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stderr
    assert "::warning file=service/src/train.py,line=2" in result.stdout
    sarif = json.loads((workspace / "repro-lens.sarif").read_text())
    assert len(sarif["runs"][0]["results"]) == 2
