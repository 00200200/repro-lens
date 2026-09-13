"""Train XGBoost in separate project copies and compare the recorded outputs."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from repro_lens.comparison import compare_reports
from repro_lens.verify import verify

FIXTURE = Path(__file__).parent / "experiment"
FILES = ("train.py", "config.toml", "samples.csv", "pyproject.toml", "uv.lock")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".repro-lens/xgboost-demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output)).resolve()
    baseline = None
    summary = {}
    print(f"{'Scenario':16} {'verify':10} {'compare':10} training RMSE", flush=True)
    for name, expected, changed in [
        ("baseline", None, []),
        ("refactor", "matched", ["train.py"]),
        ("shallower tree", "mismatch", ["config.toml"]),
    ]:
        project = directory / name.replace(" ", "-")
        project.mkdir()
        for filename in FILES:
            (project / filename).write_bytes((FIXTURE / filename).read_bytes())
        if name == "refactor":
            source = project / "train.py"
            source.write_text(
                source.read_text(encoding="utf-8").replace(
                    "features = values[:, :2]", "features = values[:, [0, 1]]"
                ),
                encoding="utf-8",
            )
        elif name == "shallower tree":
            config = project / "config.toml"
            config.write_text(
                config.read_text(encoding="utf-8").replace("max_depth = 2", "max_depth = 1"),
                encoding="utf-8",
            )
        # Keep both setup and replay inside this copy, even with a caller's uv overrides.
        os.environ["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
        os.environ.pop("VIRTUAL_ENV", None)
        subprocess.run(["uv", "sync", "--project", str(project), "--locked"], check=True)
        report = verify(project)
        if report["status"] != "matched":
            raise RuntimeError(f"Replay failed; inspect {report['report_path']}")
        summary[name] = {"report": report["report_path"]}
        if baseline is None:
            baseline = Path(report["report_path"])
        else:
            comparison = compare_reports(baseline, Path(report["report_path"]))
            path = directory / f"{name.replace(' ', '-')}-comparison.json"
            path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
            if comparison["status"] != expected or comparison["input_changes"] != {
                "added": [],
                "removed": [],
                "modified": changed,
            }:
                raise RuntimeError(f"Unexpected comparison; inspect {path}")
            summary[name]["comparison"] = str(path)
        rmse = report["runs"][0]["metrics"]["training_rmse"]
        print(f"{name:16} {report['status']:10} {expected or '-':10} {rmse:.6f}", flush=True)
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Evidence: {path}")


if __name__ == "__main__":
    main()
