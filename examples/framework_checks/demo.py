"""Scan before/after source strings without importing ML frameworks or running training."""

import argparse
import json
from pathlib import Path

from repro_lens.analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", type=Path, default=Path(".repro-lens/framework-demo.json"))
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    results = []
    print(f"{'Scenario':27} {'before':18} after")
    for case in cases:
        before, _ = analyze(case["before"], f"{case['name']}/before.py")
        after, _ = analyze(case["after"], f"{case['name']}/after.py")
        correct = [(f.code, f.severity) for f in before] == [tuple(case["expected"])] and not after
        results.append(
            {
                "name": case["name"],
                "before": [f.to_dict() for f in before],
                "after": [f.to_dict() for f in after],
                "expected_behavior": correct,
            }
        )
        status = ", ".join(f"{f.code} {f.severity}" for f in before) or "no finding"
        print(
            f"{case['name']:27} {status:18} {'no finding' if not after else 'unexpected finding'}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "assurance": "Synthetic static examples only; no frameworks imported "
                "or training executed.",
                "cases": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "\nNo finding means no covered risk detected, not proven repeatability."
        f"\nEvidence: {args.output}"
    )
    return 0 if results and all(case["expected_behavior"] for case in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
