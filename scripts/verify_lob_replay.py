"""Check raw-CSV inference against every exported held-out integer test vector."""

import argparse
import hashlib
import json
from pathlib import Path

from models.lob.data import Standardizer, prepare, read_snapshots
from models.lob.inference import LobPredictor
from models.lob.metrics import classification_metrics
from models.lob.walk_forward import final_splits, plan
from models.quantization.calibration import quantize_value


def verify(model: Path, source: Path) -> dict:
    predictor = LobPredictor(model)
    report = json.loads((model / "report.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != report["source_sha256"]:
        raise ValueError("replay verification requires the original dataset CSV")
    rows = read_snapshots(source, predictor.config.levels)
    protocol = predictor.network.get("evaluation_protocol", "single_split")
    if protocol == "walk_forward_v1":
        if predictor.network.get("walk_forward_plan") != plan(len(rows)):
            raise ValueError("Walk-forward plan does not match source data")
        splits = final_splits(rows, predictor.config)
    elif protocol == "single_split":
        splits = prepare(rows, predictor.config)
    else:
        raise ValueError("Unknown LOB evaluation protocol")
    stored_splits = json.loads((model / "split.json").read_text())
    for name, split in splits.items():
        expected_split = {"raw_start": split["raw_start"], "raw_stop": split["raw_stop"],
                          "indices": [{key: e[key] for key in ("feature_start", "decision_index", "label_end")}
                                      for e in split["examples"]]}
        if stored_splits.get(name) != expected_split or report["samples"][name] != len(split["examples"]):
            raise ValueError(f"Exported {name} split does not match chronological source data")
    expected_scaler = Standardizer.fit([e["features"] for e in splits["train"]["examples"]])
    if predictor.scaler != expected_scaler:
        raise ValueError("Saved preprocessing does not match training-only normalization")
    vectors = json.loads((model / "vectors.json").read_text())["vectors"]
    test_examples = splits["test"]["examples"]
    if len(vectors) != len(test_examples):
        raise ValueError("Held-out vector coverage is incomplete")
    predictions = []
    for vector, example in zip(vectors, test_examples):
        index = example["decision_index"]
        if (vector["dataset_index"], vector["timestamp_ns"], vector["label"]) != (
                index, example["timestamp_ns"], example["label"]):
            raise ValueError(f"Wrong held-out index, timestamp, or label at snapshot {index}")
        inputs = [quantize_value(v, predictor.input_exponent)
                  for v in predictor.scaler.transform(example["features"])]
        hidden_acc, hidden = predictor.first.evaluate(inputs)
        output_acc, logits = predictor.second.evaluate(hidden)
        if (inputs, hidden_acc, hidden, output_acc, logits) != (
                vector["inputs"], vector["hidden_accumulators"], vector["hidden"],
                vector["output_accumulators"], vector["outputs"]):
            raise ValueError(f"Input/intermediate/output mismatch at snapshot {index}")
        prediction = predictor.predict(rows, index)
        if prediction["logits"] != vector["outputs"]:
            raise AssertionError(f"Raw feature/preprocessing/export mismatch at snapshot {index}")
        predictions.append(prediction["class_id"])
    metrics = classification_metrics([e["label"] for e in test_examples], predictions)
    if metrics != report["test"]["int8_mlp"]:
        raise ValueError("Reported INT8 metrics do not match source labels and predictions")
    # Inference also works at the end of the file, where no future label is available.
    tail = predictor.predict(rows, len(rows) - 1)
    result = {"dataset_kind": report["dataset_kind"], "samples": len(vectors),
              "raw_csv_matches_export": True, "labels_and_splits_verified": True,
              "training_only_preprocessing_verified": True, "all_intermediates_verified": True,
              "int8_metrics_verified": metrics, "tail_prediction_without_future_label": tail}
    (model / "replay_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/lob"))
    parser.add_argument("--input", type=Path, default=Path("build/lob-data/snapshots.csv"))
    args = parser.parse_args()
    result = verify(args.model, args.input)
    print(f"PASS: raw CSV inference matches all {result['samples']} held-out exported predictions")


if __name__ == "__main__":
    main()
