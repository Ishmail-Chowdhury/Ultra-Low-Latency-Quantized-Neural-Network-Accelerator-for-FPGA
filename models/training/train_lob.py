"""Train, select on validation, and export a compact LOB MLP with INT8 reference outputs."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from models.lob.data import CLASSES, Standardizer
from models.lob.metrics import classification_metrics
from models.quantization.calibration import quantize_dense, quantize_value, scale_exponent
from models.quantization.dense_reference import export_model


def train(dataset: Path, output: Path, seed: int = 2026) -> dict:
    document = json.loads(dataset.read_text())
    if document.get("schema_version") != 1 or document.get("dataset_kind") not in ("synthetic", "real"):
        raise ValueError("unsupported prepared dataset")
    examples = {name: split["examples"] for name, split in document["splits"].items()}
    scaler = Standardizer.fit([e["features"] for e in examples["train"]])
    x = {name: np.asarray([scaler.transform(e["features"]) for e in rows]) for name, rows in examples.items()}
    y = {name: [e["label"] for e in rows] for name, rows in examples.items()}
    if set(y["train"]) != {0, 1, 2}:
        raise ValueError("training data must contain down, stationary, and up labels")

    candidates = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for hidden in (8, 16):
            model = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                                  alpha=1.0, batch_size=128, learning_rate_init=0.001,
                                  shuffle=False, early_stopping=False, n_iter_no_change=20,
                                  max_iter=1000, random_state=seed)
            model.fit(x["train"], y["train"])
            metrics = classification_metrics(y["validation"], model.predict(x["validation"]).tolist())
            candidates.append((metrics["macro_f1"], model, metrics))
        # max keeps the first (smaller) model on exact score ties; test data is not consulted.
        _, model, validation_metrics = max(candidates, key=lambda item: item[0])
        float_predictions = model.predict(x["test"]).tolist()
        logistic = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        logistic.fit(x["train"], y["train"])
        logistic_predictions = logistic.predict(x["test"]).tolist()
        train_hidden = np.maximum(0, x["train"] @ model.coefs_[0] + model.intercepts_[0])
        train_logits = train_hidden @ model.coefs_[1] + model.intercepts_[1]

    input_exponent = scale_exponent(float(np.max(np.abs(x["train"]))))
    hidden_exponent = scale_exponent(float(np.max(np.abs(train_hidden))))
    output_exponent = scale_exponent(float(np.max(np.abs(train_logits))))
    first, first_scales = quantize_dense(model.coefs_[0].T.tolist(), model.intercepts_[0].tolist(),
                                         input_exponent, hidden_exponent, True)
    second, second_scales = quantize_dense(model.coefs_[1].T.tolist(), model.intercepts_[1].tolist(),
                                           hidden_exponent, output_exponent, False)
    vectors = []
    input_clipping = 0
    for example, values in zip(examples["test"], x["test"]):
        inputs = [quantize_value(float(value), input_exponent) for value in values]
        input_clipping += sum(v < -128 * 2.0 ** input_exponent or v > 127 * 2.0 ** input_exponent for v in values)
        hidden_acc, hidden = first.evaluate(inputs)
        output_acc, logits = second.evaluate(hidden)
        vectors.append({"dataset_index": example["decision_index"], "timestamp_ns": example["timestamp_ns"],
                        "label": example["label"], "inputs": inputs, "hidden_accumulators": hidden_acc,
                        "hidden": hidden, "output_accumulators": output_acc, "outputs": logits})
    quant_predictions = [max(range(3), key=lambda i: vector["outputs"][i]) for vector in vectors]
    float_metrics = classification_metrics(y["test"], float_predictions)
    quant_metrics = classification_metrics(y["test"], quant_predictions)
    majority = max(range(3), key=y["train"].count)

    output.mkdir(parents=True, exist_ok=True)
    export_model(first, output / "layer0")
    export_model(second, output / "layer1")
    network = {"schema_version": 1, "task": "lob_direction", "dataset_kind": document["dataset_kind"],
               "architecture": [first.inputs, first.outputs, second.outputs], "classes": CLASSES, "seed": seed,
               "feature_config": document["config"], "feature_names": document["feature_names"],
               "input_preprocessing": "causal feature windows; training-only mean/std; symmetric INT8",
               "layers": [{"directory": "layer0", **first_scales}, {"directory": "layer1", **second_scales}]}
    split_metadata = {name: {"raw_start": document["splits"][name]["raw_start"],
                             "raw_stop": document["splits"][name]["raw_stop"],
                             "indices": [{key: e[key] for key in ("feature_start", "decision_index", "label_end")}
                                         for e in rows]} for name, rows in examples.items()}
    files = {"network.json": network, "preprocessing.json": asdict(scaler), "split.json": split_metadata,
             "float_parameters.json": {"weights": [w.T.tolist() for w in model.coefs_],
                                       "biases": [b.tolist() for b in model.intercepts_]},
             "vectors.json": {"schema_version": 1, "dataset_kind": document["dataset_kind"], "vectors": vectors}}
    for name, content in files.items():
        (output / name).write_text(json.dumps(content, indent=2) + "\n")
    artifact_names = list(files) + [f"layer{i}/{name}" for i in (0, 1)
                                    for name in ("model.json", "weights.mem", "biases.mem")]
    report = {"schema_version": 1, "task": "lob_direction", "dataset_kind": document["dataset_kind"],
              "provenance": document["provenance"], "seed": seed, "architecture": network["architecture"],
              "source_sha256": document["source_sha256"],
              "prepared_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "samples": {name: len(rows) for name, rows in examples.items()},
              "normalization_and_calibration_split": "train only",
              "selection_metric": "validation macro_f1; smaller model on exact ties",
              "selected_hyperparameters": model.get_params(),
              "candidates": [{"hidden_neurons": candidate.hidden_layer_sizes[0],
                               "iterations": int(candidate.n_iter_), "validation": metrics}
                              for _, candidate, metrics in candidates],
              "selected_validation": validation_metrics,
              "test": {"float_mlp": float_metrics, "int8_mlp": quant_metrics,
                       "majority_train_class": classification_metrics(y["test"], [majority] * len(y["test"])),
                       "always_stationary": classification_metrics(y["test"], [1] * len(y["test"])),
                       "logistic_regression": classification_metrics(y["test"], logistic_predictions)},
              "accuracy_drop_percentage_points": 100 * (float_metrics["accuracy"] - quant_metrics["accuracy"]),
              "macro_f1_drop": float_metrics["macro_f1"] - quant_metrics["macro_f1"],
              "test_input_values_outside_int8_range": int(input_clipping),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__},
              "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifact_names}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("build/lob-data/prepared/dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("build/lob"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = train(args.dataset, args.output, args.seed)
    print(f"Dataset: {report['dataset_kind'].upper()}; selected architecture: {report['architecture']}")
    for name, metrics in report["test"].items():
        print(f"{name}: accuracy={metrics['accuracy']:.4%}, macro-F1={metrics['macro_f1']:.4f}")
    print(f"Exported {report['samples']['test']} held-out vectors to {args.output}")


if __name__ == "__main__":
    main()
