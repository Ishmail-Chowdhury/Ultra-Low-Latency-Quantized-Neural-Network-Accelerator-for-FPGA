"""Verify every exported held-out vector through both RTL layers and the full MLP."""

import argparse
import json
from pathlib import Path

from models.quantization.artifacts import validate_export
from tb.dense_runner import verify_dense, verify_mlp


def verify(directory: Path) -> dict:
    report, network, first, second = validate_export(directory)
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
               "saved_memory_files_tested": True, "architecture": network["architecture"], "cases": {}}
    for stalls in [False, True]:
        suffix = "stalls" if stalls else "no_stalls"
        for index, (label, spec, inputs) in enumerate([("hidden", first, [v["inputs"] for v in vectors]),
                                    ("output", second, [v["hidden"] for v in vectors])]):
            results["cases"][f"{label}_{suffix}"] = verify_dense(
                spec, inputs, stalls=stalls, memory_directory=directory / f"layer{index}")
        results["cases"][f"mlp_{suffix}"] = verify_mlp(
            first, second, [v["inputs"] for v in vectors], stalls=stalls, memory_directory=directory)
    (directory / "rtl_report.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/digits"))
    args = parser.parse_args()
    result = verify(args.model)
    print(f"PASS: {result['samples']} held-out samples match both RTL layers and the complete MLP, with and without stalls")


if __name__ == "__main__":
    main()
