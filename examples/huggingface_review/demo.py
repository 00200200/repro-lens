"""Run static screening and comparison on Hugging Face training scripts."""

import sys
from pathlib import Path

from repro_lens.analysis import analyze

HERE = Path(__file__).resolve().parent


def main() -> int:
    baseline_path = HERE / "train_baseline.py"
    reproducible_path = HERE / "train_reproducible.py"

    print("Analyzing Hugging Face baseline vs reproducible fine-tuning scripts...\n")

    baseline_code = baseline_path.read_text(encoding="utf-8")
    reproducible_code = reproducible_path.read_text(encoding="utf-8")

    baseline_findings, _ = analyze(baseline_code, "train_baseline.py")
    reproducible_findings, _ = analyze(reproducible_code, "train_reproducible.py")

    print(f"{'Script':24} {'Findings':10} Status")
    print("-" * 50)

    # 1. Verify baseline has expected reproducibility warnings
    baseline_codes = {f.code for f in baseline_findings}
    expected_codes = {"R117", "R127"}
    if not expected_codes.issubset(baseline_codes):
        print(
            f"FAILED: Expected {expected_codes} in baseline, found {baseline_codes}",
            file=sys.stderr,
        )
        return 1
    detected = ", ".join(sorted(baseline_codes))
    print(f"{'train_baseline.py':24} {len(baseline_findings):<10} ⚠️ Detected {detected}")
    for finding in baseline_findings:
        print(f"   ↳ [{finding.code} · {finding.severity}] line {finding.line}: {finding.message}")

    # 2. Verify reproducible version has zero findings
    if reproducible_findings:
        count = len(reproducible_findings)
        print(
            f"FAILED: Expected 0 findings in reproducible version, found {count}",
            file=sys.stderr,
        )
        return 1
    print(f"\n{'train_reproducible.py':24} {0:<10} ✅ Fully Reproducible (0 findings)")

    print("\nSummary: Hugging Face fine-tuning script successfully reviewed and hardened.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
