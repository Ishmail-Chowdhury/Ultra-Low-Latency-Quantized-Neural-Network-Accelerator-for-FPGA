"""Train a 64→16→10 MLP on sklearn's bundled digits data; export INT8 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from models.quantization.calibration import quantize_dense, quantize_value, scale_exponent
from models.quantization.dense_reference import export_model


def train(output: Path, seed: int = 2026) -> dict:
    data = load_digits()
    x = data.data.astype(np.float64) / 16.0
    y = data.target
    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    model = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="lbfgs",
                          alpha=0.01, max_iter=1000, max_fun=50000, random_state=seed)
    # A small CPU model keeps the first benchmark independent of GPU/PyTorch setup.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
        float_accuracy = float(model.score(x_test, y_test))
        calibration_hidden = np.maximum(0, x_train @ model.coefs_[0] + model.intercepts_[0])
        calibration_logits = calibration_hidden @ model.coefs_[1] + model.intercepts_[1]
        float_hidden = np.maximum(0, x_test @ model.coefs_[0] + model.intercepts_[0])
        float_logits = float_hidden @ model.coefs_[1] + model.intercepts_[1]

    # Pixel values are multiples of 1/16; a 1/64 scale encodes them exactly (0..64).
    input_exponent = -6
    hidden_exponent = scale_exponent(float(np.max(np.abs(calibration_hidden))))
    output_exponent = scale_exponent(float(np.max(np.abs(calibration_logits))))
    first, first_scales = quantize_dense(model.coefs_[0].T.tolist(), model.intercepts_[0].tolist(),
                                         input_exponent, hidden_exponent, True)
    second, second_scales = quantize_dense(model.coefs_[1].T.tolist(), model.intercepts_[1].tolist(),
                                           hidden_exponent, output_exponent, False)
    vectors = []
    for index, values, label in zip(test_indices, x_test, y_test):
        integer_inputs = [quantize_value(float(v), input_exponent) for v in values]
        hidden_acc, hidden = first.evaluate(integer_inputs)
        logits_acc, logits = second.evaluate(hidden)
        vectors.append({"dataset_index": int(index), "label": int(label), "inputs": integer_inputs,
                        "hidden_accumulators": hidden_acc, "hidden": hidden,
                        "output_accumulators": logits_acc, "outputs": logits})
    predictions = np.argmax([v["outputs"] for v in vectors], axis=1)
    quant_accuracy = float(np.mean(predictions == y_test))
    accuracy_drop = float_accuracy - quant_accuracy
    error = {}
    for name, reference, values, exponent in [
        ("hidden", float_hidden, [v["hidden"] for v in vectors], hidden_exponent),
        ("logits", float_logits, [v["outputs"] for v in vectors], output_exponent),
    ]:
        absolute_error = np.abs(reference - np.asarray(values) * (2.0 ** exponent))
        error[name] = {"mean_absolute_error": float(np.mean(absolute_error)),
                       "max_absolute_error": float(np.max(absolute_error))}

    output.mkdir(parents=True, exist_ok=True)
    export_model(first, output / "layer0")
    export_model(second, output / "layer1")
    network = {"schema_version": 1, "architecture": [64, 16, 10], "seed": seed,
               "input_preprocessing": "pixels / 16; round to scale 2^-6",
               "classes": model.classes_.tolist(),
               "layers": [{"directory": "layer0", **first_scales},
                          {"directory": "layer1", **second_scales}]}
    float_parameters = {"weights": [w.T.tolist() for w in model.coefs_],
                        "biases": [b.tolist() for b in model.intercepts_]}
    split = {"train_indices": train_indices.tolist(), "test_indices": test_indices.tolist()}
    files = {"network.json": network, "float_parameters.json": float_parameters,
             "split.json": split, "vectors.json": {"schema_version": 1, "vectors": vectors}}
    for name, content in files.items():
        (output / name).write_text(json.dumps(content, indent=2) + "\n")

    report = {
        "schema_version": 1, "seed": seed, "dataset": "sklearn.datasets.load_digits",
        "dataset_source": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "architecture": [64, 16, 10], "train_samples": len(train_indices),
        "test_samples": len(test_indices), "calibration_split": "training only",
        "float_accuracy": float_accuracy, "int8_accuracy": quant_accuracy,
        "accuracy_drop_percentage_points": accuracy_drop * 100,
        "acceptance": {"minimum_float_accuracy": 0.90, "maximum_accuracy_drop_percentage_points": 2.0,
                       "passed": float_accuracy >= 0.90 and accuracy_drop <= 0.02},
        "layer_error": error, "training_iterations": int(model.n_iter_),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "sha256": {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(output.rglob("*")) if path.is_file() and path.name != "report.json"
                   and path.name != "rtl_report.json"},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["acceptance"]["passed"]:
        raise RuntimeError(f"Baseline acceptance failed; inspect {output / 'report.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/digits"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = train(args.output, args.seed)
    print(f"Float accuracy: {report['float_accuracy']:.4%}")
    print(f"INT8 accuracy:  {report['int8_accuracy']:.4%}")
    print(f"Accuracy drop:  {report['accuracy_drop_percentage_points']:.4f} percentage points")
    print(f"Exported model and {report['test_samples']} held-out vectors to {args.output}")


if __name__ == "__main__":
    main()
