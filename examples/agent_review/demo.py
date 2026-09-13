"""Run known edits in separate copies and compare their retained evidence."""

import argparse
import json
import tempfile
from pathlib import Path

from repro_lens.comparison import compare_reports
from repro_lens.verify import verify

FIXTURE = Path(__file__).parent / "experiment"


def replay(destination, code, policy, expected_accuracy):
    destination.mkdir()
    (destination / "experiment.py").write_text(code, encoding="utf-8")
    (destination / "pyproject.toml").write_text(policy, encoding="utf-8")
    (destination / "samples.csv").write_bytes((FIXTURE / "samples.csv").read_bytes())
    report = verify(destination)
    if report["status"] != "matched":
        raise RuntimeError(f"Unexpected replay result; inspect {report['report_path']}")
    if any(run["metrics"]["accuracy"] != expected_accuracy for run in report["runs"]):
        raise RuntimeError(f"Unexpected fixture accuracy; inspect {report['report_path']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".repro-lens/agent-demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output)).resolve()
    code = (FIXTURE / "experiment.py").read_text(encoding="utf-8")
    policy = (FIXTURE / "pyproject.toml").read_text(encoding="utf-8")
    changed_code = code.replace("THRESHOLD = 0.5", "THRESHOLD = 0.7")
    scenarios = [
        (
            "refactor",
            code.replace("return int(score >= THRESHOLD)", "return 1 if score >= THRESHOLD else 0"),
            policy,
            "matched",
            1.0,
            ["experiment.py"],
        ),
        ("changed threshold", changed_code, policy, "mismatch", 0.75, ["experiment.py"]),
        (
            "changed tolerance",
            changed_code,
            policy.replace("atol = 0.0", "atol = 0.3"),
            "not_comparable",
            0.75,
            ["experiment.py", "pyproject.toml"],
        ),
    ]
    baseline = replay(directory / "baseline", code, policy, 1.0)
    summary = {"baseline": baseline["report_path"], "comparisons": {}}
    print(f"{'Scenario':20} {'verify':10} compare with baseline")
    print(f"{'baseline':20} {'matched':10} -")
    for name, candidate_code, candidate_policy, expected, accuracy, modified in scenarios:
        report = replay(
            directory / name.replace(" ", "-"), candidate_code, candidate_policy, accuracy
        )
        comparison = compare_reports(Path(baseline["report_path"]), Path(report["report_path"]))
        path = directory / f"{name.replace(' ', '-')}-comparison.json"
        path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if comparison["status"] != expected or comparison["input_changes"] != {
            "added": [],
            "removed": [],
            "modified": modified,
        }:
            raise RuntimeError(f"Unexpected comparison result; inspect {path}")
        summary["comparisons"][name] = {"path": str(path), "status": comparison["status"]}
        print(f"{name:20} {report['status']:10} {comparison['status']}")
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nEvidence: {path}")


if __name__ == "__main__":
    main()
