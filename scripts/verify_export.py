"""Verify every exported held-out vector through both RTL layers and the full MLP."""

import argparse
import json
from pathlib import Path

from models.quantization.artifacts import validate_export
from tb.dense_runner import verify_dense, verify_mlp


def verify(directory: Path, *, hidden_lanes: int = 1, output_lanes: int = 1,
           report_path: Path | None = None) -> dict:
    report, network, first, second = validate_export(directory)
    if not 1 <= hidden_lanes <= first.outputs or not 1 <= output_lanes <= second.outputs:
        raise ValueError("Lane counts must be between 1 and the corresponding layer output count")
    vectors = json.loads((directory / "vectors.json").read_text())["vectors"]
    expected_count = report.get("samples", {}).get("test", report.get("test_samples"))
    if not vectors or len(vectors) != expected_count:
        raise ValueError("Held-out vector count does not match training report")
    # Check stored expectations independently before using them for RTL comparison.
    for vector in vectors:
        hidden_acc, hidden = first.evaluate(vector["inputs"])
        output_acc, outputs = second.evaluate(hidden)
        if (hidden_acc, hidden, output_acc, outputs) != (
                vector["hidden_accumulators"], vector["hidden"],
                vector["output_accumulators"], vector["outputs"]):
            raise ValueError("Stored vectors do not match the exported model")
    results = {"schema_version": 1, "samples": len(vectors), "bit_exact": True,
               "saved_memory_files_tested": True, "architecture": network["architecture"],
               "hidden_lanes": hidden_lanes, "output_lanes": output_lanes,
               "mac_instances": hidden_lanes + output_lanes, "cases": {}}
    for stalls in [False, True]:
        suffix = "stalls" if stalls else "no_stalls"
        for index, (label, spec, inputs) in enumerate([("hidden", first, [v["inputs"] for v in vectors]),
                                    ("output", second, [v["hidden"] for v in vectors])]):
            results["cases"][f"{label}_{suffix}"] = verify_dense(
                spec, inputs, stalls=stalls, memory_directory=directory / f"layer{index}",
                lanes=hidden_lanes if index == 0 else output_lanes)
        results["cases"][f"mlp_{suffix}"] = verify_mlp(
            first, second, [v["inputs"] for v in vectors], stalls=stalls, memory_directory=directory,
            hidden_lanes=hidden_lanes, output_lanes=output_lanes)
    if report_path is None:
        filename = "rtl_report.json" if (hidden_lanes, output_lanes) == (1, 1) else f"rtl_report_{hidden_lanes}x{output_lanes}.json"
        report_path = directory / filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/digits"))
    parser.add_argument("--hidden-lanes", type=int, default=1)
    parser.add_argument("--output-lanes", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.model, hidden_lanes=args.hidden_lanes, output_lanes=args.output_lanes,
                    report_path=args.report)
    print(f"PASS: {result['samples']} held-out samples match both RTL layers and the complete MLP, with and without stalls")


if __name__ == "__main__":
    main()
