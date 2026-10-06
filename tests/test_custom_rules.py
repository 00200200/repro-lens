from __future__ import annotations

import json
from pathlib import Path

from repro_lens.analysis import CustomRule, analyze
from repro_lens.project import (
    _fallback_parse_rules_yaml,
    add_ignores,
    check,
    parse_custom_rules_yaml,
)


def test_parse_custom_rules_yaml():
    yaml_content = """
rules:
  - id: C101
    severity: warning
    message: "company_ml.split requires explicit seed argument"
    suggestion: "Pass explicit seed=..."
    match:
      call: "company_ml.split"
      missing_kwargs: ["seed"]
  - id: C102
    severity: error
    message: "company_ml.data.load requires explicit seed argument"
    match:
      call: "company_ml.data.load"
      missing_kwargs: ["seed", "random_state"]
"""
    rules = parse_custom_rules_yaml(yaml_content)
    assert len(rules) == 2
    r1, r2 = rules[0], rules[1]

    assert r1.id == "C101"
    assert r1.severity == "warning"
    assert r1.message == "company_ml.split requires explicit seed argument"
    assert r1.suggestion == "Pass explicit seed=..."
    assert r1.call == "company_ml.split"
    assert r1.missing_kwargs == ("seed",)

    assert r2.id == "C102"
    assert r2.severity == "error"
    assert r2.call == "company_ml.data.load"
    assert r2.missing_kwargs == ("seed", "random_state")


def test_fallback_yaml_parser():
    yaml_content = """
rules:
  - id: C101
    severity: warning
    message: "company_ml.split requires explicit seed argument"
    match:
      call: "company_ml.split"
      missing_kwargs: ["seed"]
"""
    parsed = _fallback_parse_rules_yaml(yaml_content)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "C101"
    assert parsed[0]["match"]["call"] == "company_ml.split"
    assert parsed[0]["match"]["missing_kwargs"] == ["seed"]


def test_analyze_with_custom_rule():
    rule = CustomRule(
        id="C101",
        message="company_ml.split requires explicit seed argument",
        severity="warning",
        call="company_ml.split",
        missing_kwargs=("seed",),
    )

    # 1. Missing seed -> finding
    code_missing = "import company_ml\ncompany_ml.split(data)"
    findings, suppressed = analyze(code_missing, custom_rules=[rule])
    assert len(findings) == 1
    assert findings[0].code == "C101"
    assert findings[0].severity == "warning"
    assert "company_ml.split" in findings[0].message
    assert len(suppressed) == 0

    # 2. Explicit seed provided -> no finding
    code_provided = "import company_ml\ncompany_ml.split(data, seed=42)"
    findings, suppressed = analyze(code_provided, custom_rules=[rule])
    assert len(findings) == 0

    # 3. seed=None -> finding
    code_none = "import company_ml\ncompany_ml.split(data, seed=None)"
    findings, suppressed = analyze(code_none, custom_rules=[rule])
    assert len(findings) == 1
    assert findings[0].code == "C101"

    # 4. From import resolution
    code_from_import = "from company_ml import split\nsplit(data)"
    findings, _ = analyze(code_from_import, custom_rules=[rule])
    assert len(findings) == 1
    assert findings[0].code == "C101"

    # 5. Aliased import resolution
    code_alias = "import company_ml as cml\ncml.split(data)"
    findings, _ = analyze(code_alias, custom_rules=[rule])
    assert len(findings) == 1
    assert findings[0].code == "C101"


def test_custom_rule_suppression():
    rule = CustomRule(
        id="C101",
        message="company_ml.split requires explicit seed argument",
        severity="warning",
        call="company_ml.split",
        missing_kwargs=("seed",),
    )
    code = (
        "import company_ml\n"
        "company_ml.split(data)  # repro-lens: ignore[C101] -- deterministic upstream\n"
    )
    findings, suppressed = analyze(code, custom_rules=[rule])
    assert len(findings) == 0
    assert len(suppressed) == 1
    assert suppressed[0].code == "C101"


def test_project_check_with_rules_yaml(tmp_path: Path):
    rules_dir = tmp_path / ".repro-lens"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yaml").write_text(
        """
rules:
  - id: C101
    severity: warning
    message: "company_ml.split requires explicit seed argument"
    match:
      call: "company_ml.split"
      missing_kwargs: ["seed"]
""",
        encoding="utf-8",
    )

    # File with violation
    (tmp_path / "train.py").write_text(
        "import company_ml\ncompany_ml.split(train_data)\n",
        encoding="utf-8",
    )

    report = check(tmp_path)
    codes = [f["code"] for f in report["findings"]]
    assert "C101" in codes

    # Test adding ignore via add_ignores
    report_after_ignore = add_ignores(tmp_path)
    codes_after = [f["code"] for f in report_after_ignore["findings"]]
    assert "C101" not in codes_after
    suppressed_codes = [f["code"] for f in report_after_ignore["suppressed"]]
    assert "C101" in suppressed_codes


def test_notebook_with_custom_rule(tmp_path: Path):
    rules_dir = tmp_path / ".repro-lens"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yaml").write_text(
        """
rules:
  - id: C101
    severity: warning
    message: "company_ml.split requires explicit seed argument"
    match:
      call: "company_ml.split"
      missing_kwargs: ["seed"]
""",
        encoding="utf-8",
    )

    nb_data = {
        "cells": [
            {
                "cell_type": "code",
                "source": "import company_ml\ncompany_ml.split(data)\n",
                "metadata": {},
                "outputs": [],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    (tmp_path / "experiment.ipynb").write_text(json.dumps(nb_data), encoding="utf-8")

    report = check(tmp_path)
    assert any(f["code"] == "C101" for f in report["findings"])
