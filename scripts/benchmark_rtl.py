"""Compare parallel RTL configurations on the same saved model and test vectors."""

import argparse
import hashlib
import json
from pathlib import Path

from models.quantization.artifacts import validate_export
from scripts.verify_export import verify


def lane_pair(value: str) -> tuple[int, int]:
    try:
        hidden, output = map(int, value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use HIDDEN:OUTPUT, for example 4:3") from exc
    if hidden < 1 or output < 1:
        raise argparse.ArgumentTypeError("Lane counts must be positive")
    return hidden, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/lob-improved/model"))
    parser.add_argument("--output", type=Path, default=Path("build/rtl-parallel"))
    parser.add_argument("--lanes", type=lane_pair, action="append",
                        help="Repeat HIDDEN:OUTPUT pairs; the 1:1 baseline is always included")
    args = parser.parse_args()
    _, network, first, second = validate_export(args.model)
    pairs = list(dict.fromkeys([(1, 1)] + (args.lanes or [(2, 1), (4, 1), (4, 3), (8, 3)])))
    if any(hidden > first.outputs or output > second.outputs for hidden, output in pairs):
        parser.error("Lane counts cannot exceed the corresponding layer output count")
    files = ["network.json", "vectors.json"] + [
        f"layer{index}/{name}" for index in (0, 1) for name in ("model.json", "weights.mem", "biases.mem")]
    hashes = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest() for name in files}
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "status": "running", "model": str(args.model),
               "architecture": network["architecture"], "artifact_sha256": hashes,
               "measurement": "RTL simulation clock periods; no synthesis or hardware timing",
               "configurations": []}
    summary_path = args.output / "report.json"
    # A failed rerun must not leave an earlier successful summary at this path.
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    baseline_cycles = None
    for hidden, output in pairs:
        name = f"lanes-{hidden}x{output}.json"
        report = verify(args.model, hidden_lanes=hidden, output_lanes=output,
                        report_path=args.output / name)
        case = report["cases"]["mlp_no_stalls"]
        cycles = case["max_last_input_to_last_output_cycles"]
        if baseline_cycles is None:
            baseline_cycles = cycles
        summary["configurations"].append({
            "hidden_lanes": hidden, "output_lanes": output, "mac_instances": hidden + output,
            "samples": report["samples"], "bit_exact": report["bit_exact"],
            "min_latency_cycles": case["min_last_input_to_last_output_cycles"],
            "max_latency_cycles": cycles,
            "min_input_start_interval_cycles": case["min_input_start_interval_cycles"],
            "max_input_start_interval_cycles": case["max_input_start_interval_cycles"],
            "baseline_to_current_max_latency_ratio": baseline_cycles / cycles,
            "verification_report": name,
        })
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"PASS {hidden}:{output}: {cycles} max no-stall latency cycles; "
              f"{report['samples']} samples bit-exact with and without stalls", flush=True)
    if any(hashlib.sha256((args.model / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise ValueError("Model artifacts changed during the benchmark")
    summary["status"] = "passed"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
