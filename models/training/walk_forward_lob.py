"""Select class balancing and model size on expanding chronological validation folds."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from models.lob.data import CLASSES, FeatureConfig, Standardizer, feature_names, read_snapshots
from models.lob.metrics import classification_metrics
from models.lob.walk_forward import balanced_weights, final_splits, plan, select_candidate, selection_folds
from models.quantization.calibration import quantize_dense, quantize_value, scale_exponent
from models.quantization.dense_reference import export_model


# Order is fixed before observing results; it also defines the exact-tie policy.
CANDIDATES = [
    {"name": "mlp_8", "family": "mlp", "hidden": 8, "balanced": False},
    {"name": "mlp_8_balanced", "family": "mlp", "hidden": 8, "balanced": True},
    {"name": "mlp_16", "family": "mlp", "hidden": 16, "balanced": False},
    {"name": "mlp_16_balanced", "family": "mlp", "hidden": 16, "balanced": True},
    {"name": "logistic", "family": "logistic", "balanced": False},
    {"name": "logistic_balanced", "family": "logistic", "balanced": True},
]


def candidates_for(profile):
    if profile == "baseline":
        return CANDIDATES
    if profile != "stationary":
        raise ValueError("unknown experiment profile")
    return [{"name": f"mlp_{hidden}_alpha{alpha}_offset{offset}", "family": "mlp",
             "hidden": hidden, "balanced": True, "alpha": alpha, "stationary_offset": offset}
            for hidden in (8, 16) for alpha in (1.0, 5.0) for offset in (0.0, 0.125, 0.25, 0.5, 0.75)] + [CANDIDATES[-1]]


def adjusted_model(model, offset):
    # Bake a validation-selected decision adjustment into the existing output bias.
    if not offset:
        return model
    result = copy.deepcopy(model)
    result.intercepts_[-1][1] += offset
    return result


def quantized_model(model, training_inputs):
    hidden = np.maximum(0, training_inputs @ model.coefs_[0] + model.intercepts_[0])
    logits = hidden @ model.coefs_[1] + model.intercepts_[1]
    input_exp = scale_exponent(float(np.max(np.abs(training_inputs))))
    hidden_exp = scale_exponent(float(np.max(np.abs(hidden))))
    output_exp = scale_exponent(float(np.max(np.abs(logits))))
    first, first_scales = quantize_dense(model.coefs_[0].T.tolist(), model.intercepts_[0].tolist(),
                                         input_exp, hidden_exp, True)
    second, second_scales = quantize_dense(model.coefs_[1].T.tolist(), model.intercepts_[1].tolist(),
                                           hidden_exp, output_exp, False)
    return first, second, first_scales, second_scales


def integer_scores(first, second, values, input_exponent):
    scores = []
    for row in values:
        inputs = [quantize_value(float(v), input_exponent) for v in row]
        _, hidden = first.evaluate(inputs)
        scores.append(second.evaluate(hidden)[1])
    return scores


def fit_candidate(candidate, x, y, seed):
    if set(y) != {0, 1, 2}:
        raise ValueError("each training interval must contain all three classes")
    if candidate["family"] == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(candidate["hidden"],), activation="relu", solver="adam",
                              alpha=candidate.get("alpha", 1.0), batch_size=128, learning_rate_init=0.001,
                              shuffle=False, early_stopping=False, n_iter_no_change=20,
                              max_iter=1500, random_state=seed)
    else:
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    weights = balanced_weights(y) if candidate["balanced"] else None
    model.fit(x, y, sample_weight=weights)
    return model


def scaled_split(split, scaler):
    return (np.asarray([scaler.transform(e["features"]) for e in split["examples"]]),
            [e["label"] for e in split["examples"]])


def evaluate(source: Path, output: Path, config: FeatureConfig, kind: str,
             provenance: str, seed: int = 2027, profile: str = "baseline") -> dict:
    if kind not in ("synthetic", "real") or not provenance.strip():
        raise ValueError("explicit dataset kind and provenance are required")
    rows = read_snapshots(source, config.levels)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    sidecar = source.with_suffix(".metadata.json")
    source_metadata = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if source_metadata is not None and any(source_metadata.get(k) != expected for k, expected in
            (("dataset_kind", kind), ("levels", config.levels), ("tick_size", config.tick_size), ("sha256", source_hash))):
        raise ValueError("source metadata differs from requested data/configuration")
    experiment_plan = plan(len(rows))
    records = [{**candidate, "fold_metrics": [], "training_iterations": [],
                "selection_precision": "int8" if profile == "stationary" and candidate["family"] == "mlp" else "float"}
               for candidate in candidates_for(profile)]
    fold_audit = []
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for index, fold in enumerate(selection_folds(rows, config)):
            scaler = Standardizer.fit([e["features"] for e in fold["train"]["examples"]])
            x_train, y_train = scaled_split(fold["train"], scaler)
            x_val, y_val = scaled_split(fold["validation"], scaler)
            fold_audit.append({"fold": index + 1, "training_samples": len(y_train),
                               "validation_samples": len(y_val), "scaler": asdict(scaler),
                               "training_class_counts": dict(zip(CLASSES, (y_train.count(i) for i in range(3))))})
            fitted = {}
            for record in records:
                key = (record["family"], record.get("hidden"), record["balanced"], record.get("alpha", 1.0))
                if key not in fitted:
                    fitted[key] = fit_candidate(record, x_train, y_train, seed)
                model = adjusted_model(fitted[key], record.get("stationary_offset", 0))
                if record["selection_precision"] == "int8":
                    first, second, scales, _ = quantized_model(model, x_train)
                    scores = integer_scores(first, second, x_val, scales["input_scale_exponent"])
                    predictions = [max(range(3), key=lambda i: row[i]) for row in scores]
                else:
                    predictions = model.predict(x_val).tolist()
                metrics = classification_metrics(y_val, predictions)
                record["fold_metrics"].append(metrics)
                record["training_iterations"].append(int(np.max(model.n_iter_)))
                print(f"Fold {index + 1}: {record['name']} macro-F1={metrics['macro_f1']:.4f}", flush=True)

        for record in records:
            values = [m["macro_f1"] for m in record["fold_metrics"]]
            record["mean_validation_macro_f1"] = statistics.mean(values)
            record["validation_macro_f1_std"] = statistics.pstdev(values)
        selected_mlp = select_candidate(records, "mlp")
        selected_linear = select_candidate(records, "logistic")
        selection = {"mlp": selected_mlp["name"], "logistic": selected_linear["name"], "profile": profile,
                     "rule": "maximum mean of three validation macro-F1 scores; declared candidate order breaks ties",
                     "plan": experiment_plan, "candidates": records}
        # Freeze and persist selection before constructing or evaluating final test examples.
        (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(f"Selected {selection['mlp']} and {selection['logistic']}; now evaluating final holdout", flush=True)
        splits = final_splits(rows, config)
        scaler = Standardizer.fit([e["features"] for e in splits["train"]["examples"]])
        x_train, y_train = scaled_split(splits["train"], scaler)
        x_test, y_test = scaled_split(splits["test"], scaler)
        model = adjusted_model(fit_candidate(selected_mlp, x_train, y_train, seed),
                               selected_mlp.get("stationary_offset", 0))
        linear = fit_candidate(selected_linear, x_train, y_train, seed)
        float_metrics = classification_metrics(y_test, model.predict(x_test).tolist())
        linear_metrics = classification_metrics(y_test, linear.predict(x_test).tolist())
        train_hidden = np.maximum(0, x_train @ model.coefs_[0] + model.intercepts_[0])
        train_logits = train_hidden @ model.coefs_[1] + model.intercepts_[1]
        baseline_artifact = None
        baseline_scores = None
        baseline_metrics = None
        if profile == "stationary":
            baseline = fit_candidate(CANDIDATES[1], x_train, y_train, seed)
            base_first, base_second, base_scales, base_output_scales = quantized_model(baseline, x_train)
            baseline_scores = integer_scores(base_first, base_second, x_test, base_scales["input_scale_exponent"])
            baseline_metrics = classification_metrics(y_test, [max(range(3), key=lambda i: row[i]) for row in baseline_scores])
            baseline_artifact = {"candidate": CANDIDATES[1], "layer0": base_first.to_dict(),
                                 "layer1": base_second.to_dict(), "scales": [base_scales, base_output_scales]}

    input_exponent = scale_exponent(float(np.max(np.abs(x_train))))
    hidden_exponent = scale_exponent(float(np.max(np.abs(train_hidden))))
    output_exponent = scale_exponent(float(np.max(np.abs(train_logits))))
    first, first_scales = quantize_dense(model.coefs_[0].T.tolist(), model.intercepts_[0].tolist(),
                                         input_exponent, hidden_exponent, True)
    second, second_scales = quantize_dense(model.coefs_[1].T.tolist(), model.intercepts_[1].tolist(),
                                           hidden_exponent, output_exponent, False)
    vectors = []
    for example, values in zip(splits["test"]["examples"], x_test):
        inputs = [quantize_value(float(value), input_exponent) for value in values]
        hidden_acc, hidden = first.evaluate(inputs)
        output_acc, logits = second.evaluate(hidden)
        vectors.append({"dataset_index": example["decision_index"], "timestamp_ns": example["timestamp_ns"],
                        "label": example["label"], "inputs": inputs, "hidden_accumulators": hidden_acc,
                        "hidden": hidden, "output_accumulators": output_acc, "outputs": logits})
    int_metrics = classification_metrics(y_test, [max(range(3), key=lambda i: v["outputs"][i]) for v in vectors])
    majority = max(range(3), key=y_train.count)
    export_model(first, output / "layer0")
    export_model(second, output / "layer1")
    network = {"schema_version": 1, "task": "lob_direction", "dataset_kind": kind,
               "evaluation_protocol": "walk_forward_v1", "walk_forward_plan": experiment_plan,
               "experiment_profile": profile, "stationary_logit_offset": selected_mlp.get("stationary_offset", 0),
               "architecture": [first.inputs, first.outputs, second.outputs], "classes": CLASSES, "seed": seed,
               "feature_config": asdict(config), "feature_names": feature_names(config),
               "input_preprocessing": "causal features; development-only normalization and scale calibration",
               "layers": [{"directory": "layer0", **first_scales}, {"directory": "layer1", **second_scales}]}
    split_export = {name: {"raw_start": split["raw_start"], "raw_stop": split["raw_stop"],
                          "indices": [{key: e[key] for key in ("feature_start", "decision_index", "label_end")}
                                      for e in split["examples"]]} for name, split in splits.items()}
    files = {"network.json": network, "preprocessing.json": asdict(scaler), "split.json": split_export,
             "folds.json": {"plan": experiment_plan, "folds": fold_audit},
             "float_parameters.json": {"weights": [w.T.tolist() for w in model.coefs_],
                                       "biases": [b.tolist() for b in model.intercepts_]},
             "linear_parameters.json": {"classes": linear.classes_.tolist(), "weights": linear.coef_.tolist(),
                                        "biases": linear.intercept_.tolist()},
             "vectors.json": {"schema_version": 1, "dataset_kind": kind, "vectors": vectors}}
    if baseline_artifact is not None:
        files["paired_baseline.json"] = {"model": baseline_artifact,
            "vectors": [{"dataset_index": vector["dataset_index"], "label": vector["label"], "outputs": scores}
                        for vector, scores in zip(vectors, baseline_scores)]}
    for name, content in files.items():
        (output / name).write_text(json.dumps(content, indent=2) + "\n")
    names = [*files, "selection.json"] + [f"layer{i}/{name}" for i in (0, 1)
                                           for name in ("model.json", "weights.mem", "biases.mem")]
    report = {"schema_version": 1, "task": "lob_direction", "dataset_kind": kind,
              "evaluation_protocol": "walk_forward_v1", "source_sha256": source_hash,
              "source_metadata": source_metadata, "provenance": provenance, "seed": seed,
              "architecture": network["architecture"], "selection": selection,
              "samples": {name: len(split["examples"]) for name, split in splits.items()},
              "selected_hyperparameters": model.get_params(),
              "normalization_and_calibration_split": "fold train during selection; full development prefix for final fit",
              "test": {"float_mlp": float_metrics, "int8_mlp": int_metrics,
                       "selected_logistic": linear_metrics,
                       "majority_train_class": classification_metrics(y_test, [majority] * len(y_test)),
                       "always_stationary": classification_metrics(y_test, [1] * len(y_test))},
              "accuracy_drop_percentage_points": 100 * (float_metrics["accuracy"] - int_metrics["accuracy"]),
              "macro_f1_drop": float_metrics["macro_f1"] - int_metrics["macro_f1"],
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__},
              "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}}
    if baseline_metrics is not None:
        report["test"]["previous_8_balanced_int8_same_data"] = baseline_metrics
        report["paired_comparison"] = {
            "same_data_and_training_seed": True,
            "macro_f1_delta": int_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "accuracy_delta_percentage_points": 100 * (int_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "stationary_recall_delta_percentage_points": 100 * (int_metrics["recall"]["stationary"] - baseline_metrics["recall"]["stationary"])}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/lob-walk-forward/model"))
    parser.add_argument("--kind", choices=["synthetic", "real"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--profile", choices=["baseline", "stationary"], default="baseline")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--threshold-ticks", type=float, default=0.5)
    args = parser.parse_args()
    config = FeatureConfig(args.levels, args.lookback, args.horizon, args.tick_size, args.threshold_ticks)
    report = evaluate(args.input, args.output, config, args.kind, args.source, args.seed, args.profile)
    print(f"Final held-out results on {report['dataset_kind'].upper()} data:")
    for name, metrics in report["test"].items():
        print(f"{name}: accuracy={metrics['accuracy']:.4%}, macro-F1={metrics['macro_f1']:.4f}, "
              f"stationary recall={metrics['recall']['stationary']:.4%}")


if __name__ == "__main__":
    main()
