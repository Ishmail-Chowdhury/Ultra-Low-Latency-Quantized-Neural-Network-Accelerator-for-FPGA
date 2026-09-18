"""Run fresh, reproducible software validation and retain a machine-readable audit."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def validate(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "passed": False, "hardware_required": False, "run_directory": run.name, "steps": []}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def step(name, args):
        record = {"name": name, "passed": False, "log": f"{run.name}/{name}.log"}
        report["steps"].append(record)
        save()
        print(f"RUN  {name}", flush=True)
        with (output / record["log"]).open("w") as log:
            result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300)
        record["returncode"] = result.returncode
        record["passed"] = result.returncode == 0
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {output / record['log']}")
        print(f"PASS {name}", flush=True)
        return (output / record["log"]).read_text()

    try:
        save()
        python_log = step("python_tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])
        rtl_log = step("rtl_tests", ["-m", "unittest", "discover", "-s", "tb", "-t", ".", "-v"])
        report["tests"] = {name: int(re.search(r"Ran (\d+) tests", log)[1])
                           for name, log in (("python", python_log), ("rtl", rtl_log))}
        source = run / "snapshots.csv"
        prepared = run / "prepared"
        step("generate_data", ["-m", "scripts.generate_lob", "--output", str(source)])
        step("prepare_data", ["-m", "scripts.prepare_lob", "--input", str(source),
                              "--output", str(prepared), "--kind", "synthetic",
                              "--source", "Local pressure-driven toy generator"])
        lob = run / "lob"
        repeated = run / "lob-repeat"
        for name, path in (("train_lob", lob), ("repeat_lob_training", repeated)):
            step(name, ["-m", "models.training.train_lob", "--dataset", str(prepared / "dataset.json"),
                        "--output", str(path)])
        first = json.loads((lob / "report.json").read_text())
        second = json.loads((repeated / "report.json").read_text())
        if first != second:
            raise AssertionError("Repeated training reports or exported artifact hashes differ")
        report["deterministic_training"] = True
        step("lob_rtl", ["-m", "scripts.verify_export", "--model", str(lob)])
        step("lob_source_audit", ["-m", "scripts.verify_lob_replay", "--model", str(lob), "--input", str(source)])
        predictions = lob / "predictions.csv"
        step("csv_replay", ["-m", "scripts.predict_lob", "--model", str(lob), "--input", str(source),
                            "--output", str(predictions)])
        with predictions.open(newline="") as stream:
            replay = list(csv.DictReader(stream))
        network = json.loads((lob / "network.json").read_text())
        expected_start = network["feature_config"]["lookback"] - 1
        if [int(row["decision_index"]) for row in replay] != list(range(expected_start, 3000)):
            raise AssertionError("CSV replay dropped, duplicated, or reordered predictions")
        vectors = json.loads((lob / "vectors.json").read_text())["vectors"]
        for vector in vectors:
            row = replay[vector["dataset_index"] - expected_start]
            scores = [int(row[f"logit_{label}"]) for label in ("down", "stationary", "up")]
            if (scores != vector["outputs"] or int(row["timestamp_ns"]) != vector["timestamp_ns"] or
                    int(row["class_id"]) != max(range(3), key=lambda i: scores[i])):
                raise AssertionError("Written prediction CSV does not match held-out vectors")
        digits = run / "digits"
        step("train_digits", ["-m", "models.training.train_digits", "--output", str(digits)])
        step("digits_rtl", ["-m", "scripts.verify_export", "--model", str(digits)])
        report["coverage"] = {"lob_held_out_samples": len(vectors), "csv_predictions": len(replay),
                              "digits_held_out_samples": json.loads((digits / "report.json").read_text())["test_samples"],
                              "saved_memory_files_tested": True, "labels_splits_scaler_and_metrics_audited": True}
        report["lob_test_metrics"] = first["test"]
        report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(f"PASS: software validation report: {output / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/software-validation"))
    args = parser.parse_args()
    validate(args.output)


if __name__ == "__main__":
    main()
