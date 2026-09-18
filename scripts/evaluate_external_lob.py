"""Score a frozen exported LOB model on new snapshots, optionally checking RTL.

No fitting, calibration, or hyperparameter selection uses the external data.
"""

import argparse
import json
from pathlib import Path

from models.lob.data import examples_in_range, read_snapshots
from models.lob.inference import LobPredictor
from models.lob.metrics import classification_metrics
from models.quantization.calibration import quantize_value
from scripts.fetch_itch_sample import sha256_file
from tb.dense_runner import verify_mlp


def evaluate(model, source, output, *, rtl=False, hidden_lanes=4, output_lanes=3):
    predictor = LobPredictor(model)
    metadata = json.loads(source.with_suffix(".metadata.json").read_text())
    digest = sha256_file(source)
    training_report = json.loads((model / "report.json").read_text())
    if digest == training_report.get("source_sha256"):
        raise ValueError("External evaluation source is the training source; use held-out export verification instead")
    for key, expected in (("dataset_kind", "real"), ("sha256", digest),
                          ("levels", predictor.config.levels), ("tick_size", predictor.config.tick_size)):
        if metadata.get(key) != expected:
            raise ValueError(f"External source metadata mismatch: {key}")
    rows = read_snapshots(source, predictor.config.levels)
    examples = examples_in_range(rows, predictor.config, 0, len(rows))["examples"]
    vectors, predictions = [], []
    clipped = 0
    for example in examples:
        scaled = predictor.scaler.transform(example["features"])
        clipped += sum(value / (2 ** predictor.input_exponent) < -128 or
                       value / (2 ** predictor.input_exponent) > 127 for value in scaled)
        inputs = [quantize_value(v, predictor.input_exponent) for v in scaled]
        hidden = predictor.first.evaluate(inputs)[1]
        logits = predictor.second.evaluate(hidden)[1]
        predictions.append(max(range(3), key=lambda i: logits[i]))
        vectors.append(inputs)
    labels = [e["label"] for e in examples]
    report = {"schema_version": 1, "protocol": "frozen_model_external_data_no_refit",
              "model_training_dataset_kind": predictor.network["dataset_kind"],
              "model_report_sha256": sha256_file(model / "report.json"),
              "source_metadata": metadata, "raw_rows": len(rows), "examples": len(examples),
              "first_decision_timestamp_ns": examples[0]["timestamp_ns"],
              "last_decision_timestamp_ns": examples[-1]["timestamp_ns"],
              "feature_config": predictor.network["feature_config"],
              "metrics": classification_metrics(labels, predictions),
              "always_stationary": classification_metrics(labels, [1] * len(labels)),
              "input_values_outside_int8_range_fraction": clipped / (len(examples) * predictor.first.inputs),
              "rtl_verified": False, "rtl": {}}
    if rtl:
        for stalls in (False, True):
            report["rtl"]["stalls" if stalls else "no_stalls"] = verify_mlp(
                predictor.first, predictor.second, vectors, stalls=stalls,
                memory_directory=model, hidden_lanes=hidden_lanes, output_lanes=output_lanes)
        report["rtl_verified"] = True
        report["rtl_lanes"] = {"hidden": hidden_lanes, "output": output_lanes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/lob-improved/model"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtl", action="store_true")
    parser.add_argument("--hidden-lanes", type=int, default=4)
    parser.add_argument("--output-lanes", type=int, default=3)
    args = parser.parse_args()
    result = evaluate(args.model, args.input, args.output, rtl=args.rtl,
                      hidden_lanes=args.hidden_lanes, output_lanes=args.output_lanes)
    print(json.dumps({key: result[key] for key in ("examples", "metrics", "always_stationary", "rtl_verified")}, indent=2))


if __name__ == "__main__":
    main()
