"""Develop models on chronological days; freeze selection before a separate test run."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from models.lob.data import (CLASSES, FeatureConfig, Standardizer, feature_names,
                             read_snapshots, movement_label)
from models.lob.day_split import split_for, validate_plan
from models.lob.metrics import classification_metrics
from models.lob.walk_forward import balanced_weights
from models.training.diagnostics import input_diagnostics, paired_accuracy_interval
from models.training.integer_batch import mlp_batch
from models.training.walk_forward_lob import adjusted_model, quantized_model, scaled_split
from models.quantization.dense_reference import export_model
from scripts.fetch_itch_sample import sha256_file

HORIZONS = (1, 5, 20)
OFFSETS = (0., .25, .5)
# Fixed before development results. Matched ablations separate feature, scaling,
# class-weight, width, and regularization changes. CPU comparators use full features.
CANDIDATES = [
    {"name": "legacy8", "family": "mlp", "hidden": 8, "alpha": 1., "balanced": True,
     "features": False, "scaling": "standard", "clip": None},
    *[{"name": f"original16_b{b}", "family": "mlp", "hidden": 16, "alpha": 1., "balanced": b,
       "features": False, "scaling": "standard", "clip": None} for b in (False, True)],
    *[{"name": f"features16_{method}_clip{clip}_b{b}", "family": "mlp", "hidden": 16,
       "alpha": 1., "balanced": b, "features": True, "scaling": method, "clip": clip}
      for method, clip in (("standard", None), ("standard", 4.), ("robust", 4.)) for b in (False, True)],
    *[{"name": f"features32_alpha{alpha}", "family": "mlp", "hidden": 32, "alpha": alpha,
       "balanced": False, "features": True, "scaling": "standard", "clip": 4.} for alpha in (1., 5.)],
    *[{"name": f"{family}_b{b}", "family": family, "balanced": b, "features": True,
       "scaling": "standard", "clip": 4.} for family in ("logistic", "tree") for b in (False, True)],
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_source(source):
    metadata = json.loads(source.with_suffix(".metadata.json").read_text())
    if (metadata.get("dataset_kind") != "real" or metadata.get("sha256") != sha256_file(source) or
            metadata.get("interval_ns") != 10**9 or metadata.get("tick_size") != .01 or metadata.get("levels") != 5):
        raise ValueError("Requires checked real five-level, one-second data with a $0.01 tick")
    rows = read_snapshots(source)
    validate_plan(rows, metadata["split_days"])
    return rows, metadata


def fit(candidate, x, y, seed):
    if set(y) != {0, 1, 2}:
        raise ValueError("Training must contain all three classes")
    if candidate["family"] == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(candidate["hidden"],), alpha=candidate["alpha"],
                              batch_size=256, learning_rate_init=.001, shuffle=False,
                              early_stopping=False, n_iter_no_change=12, max_iter=300, random_state=seed)
    elif candidate["family"] == "logistic":
        model = LogisticRegression(C=1., max_iter=2000, random_state=seed)
    else:
        model = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=150,
                                               l2_regularization=1., early_stopping=False, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, sample_weight=balanced_weights(y) if candidate["balanced"] else None)
    return model, [str(w.message) for w in caught]


def rank(record):
    # Different horizons have different stationary prevalence: compare excess
    # accuracy over each horizon's own baseline, not raw cross-horizon accuracy.
    return record["validation_accuracy_delta_vs_best_baseline"], record["validation"]["macro_f1"]


def predictions(bundle, x):
    if bundle["quantized"]:
        first, second, scales, _ = bundle["quantized"]
        return mlp_batch(first, second, x, scales["input_scale_exponent"])[-1].argmax(axis=1).tolist()
    return bundle["model"].predict(x).tolist()


def develop(source, output, seed=2030, synthetic_model=None):
    rows, metadata = load_source(source)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("Use a new output directory; an existing experiment must not be overwritten")
    protocol = {"schema_version": 1, "evaluation_protocol": "calendar_days_v1",
                "source_sha256": metadata["sha256"], "split_days": metadata["split_days"], "seed": seed,
                "horizons_seconds": HORIZONS, "candidates": CANDIDATES, "stationary_offsets": OFFSETS,
                "selection": "maximum validation accuracy excess over strongest horizon-specific simple baseline (stationary, train-majority, last-move persistence); macro-F1 breaks ties; declared order on exact ties",
                "label": "endpoint mid-price movement, stationary within +/-0.5 tick",
                "normalization_and_calibration": "train days only; no validation refit",
                "test_role": "separate evaluate phase after selection is frozen",
                "test_comparisons": ["selected MLP, logistic, tree each at own selected horizon",
                                     "legacy MLP at selected MLP horizon", "selected MLP at fixed five-second horizon"],
                "uncertainty": "paired 60-second block bootstrap on final day, 2000 resamples",
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
                "code_sha256": {name: sha256_file(Path(name)) for name in (
                    "models/training/train_days.py", "models/training/diagnostics.py",
                    "models/training/integer_batch.py", "models/lob/data.py", "models/lob/day_split.py",
                    "models/training/walk_forward_lob.py", "models/quantization/calibration.py")}}
    dump(output / "protocol.json", protocol)
    records, bundles, horizon_audits = [], {}, []
    with threadpool_limits(limits=1):
        for horizon in HORIZONS:
            cache = {}
            for features in (False, True):
                config = FeatureConfig(horizon=horizon, include_mid_changes=features, include_queue_features=features)
                train_split = split_for(rows, config, metadata["split_days"], "train")
                validation = split_for(rows, config, metadata["split_days"], "validation")
                raw_train = [e["features"] for e in train_split["examples"]]
                raw_val = [e["features"] for e in validation["examples"]]
                for method, clip in (("standard", None), ("standard", 4.), ("robust", 4.)) if features else (("standard", None),):
                    scaler = Standardizer.fit(raw_train, method, clip)
                    x_train, y_train = scaled_split(train_split, scaler)
                    x_val, y_val = scaled_split(validation, scaler)
                    cache[features, method, clip] = config, scaler, x_train, y_train, x_val, y_val, raw_train, raw_val
            stationary = classification_metrics(y_val, [1] * len(y_val))
            majority = max(range(3), key=y_train.count)
            momentum = [movement_label(rows[e["decision_index"] - 1].mid, rows[e["decision_index"]].mid, config)
                        for e in validation["examples"]]
            baselines = {"always_stationary": stationary,
                         "majority_train_class": classification_metrics(y_val, [majority] * len(y_val)),
                         "last_move_persistence": classification_metrics(y_val, momentum)}
            best_baseline = max(baselines, key=lambda name: baselines[name]["accuracy"])
            horizon_audits.append({"horizon_seconds": horizon, "train_samples": len(y_train),
                                   "validation_samples": len(y_val),
                                   "train_class_counts": dict(zip(CLASSES, (y_train.count(i) for i in range(3)))),
                                   "validation_stationary": stationary,
                                   "validation_baselines": baselines, "best_baseline": best_baseline,
                                   "majority_train_class_id": majority})
            if horizon == 5 and synthetic_model:
                from models.lob.inference import LobPredictor
                predictor = LobPredictor(synthetic_model)
                synthetic_train = split_for(rows, predictor.config, metadata["split_days"], "train")
                raw = [e["features"] for e in synthetic_train["examples"]]
                dump(output / "synthetic_transfer_diagnostic.json", {
                    "scope": "real TRAIN days only; diagnostic of frozen synthetic export, no refit",
                    "synthetic_report_sha256": sha256_file(synthetic_model / "report.json"),
                    "diagnostic": input_diagnostics(raw, predictor.scaler, predictor.input_exponent,
                                                    feature_names(predictor.config))})
            for candidate in CANDIDATES:
                config, scaler, x_train, y_train, x_val, y_val, raw_train, raw_val = cache[
                    candidate["features"], candidate["scaling"], candidate["clip"]]
                model, training_warnings = fit(candidate, x_train, y_train, seed)
                for offset in OFFSETS if candidate["family"] == "mlp" and candidate["name"] != "legacy8" else (0.,):
                    adjusted = adjusted_model(model, offset) if candidate["family"] == "mlp" else model
                    quantized = quantized_model(adjusted, x_train) if candidate["family"] == "mlp" else None
                    bundle = {"model": adjusted, "quantized": quantized, "config": config, "scaler": scaler}
                    pred = predictions(bundle, x_val)
                    metrics = classification_metrics(y_val, pred)
                    name = f"h{horizon}_{candidate['name']}_offset{offset}"
                    record = {**candidate, "name": name, "base_name": candidate["name"], "horizon_seconds": horizon,
                              "stationary_offset": offset, "validation": metrics,
                              "validation_accuracy_delta_vs_stationary": metrics["accuracy"] - stationary["accuracy"],
                              "validation_accuracy_delta_vs_best_baseline": metrics["accuracy"] - baselines[best_baseline]["accuracy"],
                              "float_validation": classification_metrics(y_val, adjusted.predict(x_val).tolist()),
                              "estimator_parameters": adjusted.get_params(),
                              "training_iterations": int(np.max(adjusted.n_iter_)), "training_warnings": training_warnings,
                              "selection_precision": "int8" if quantized else "float"}
                    if quantized:
                        exponent = quantized[2]["input_scale_exponent"]
                        record["validation_input_diagnostic"] = input_diagnostics(raw_val, scaler, exponent, feature_names(config))
                    records.append(record)
                    bundles[name] = bundle
                print(f"Develop h={horizon}s {candidate['name']}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}", flush=True)
                dump(output / "development-progress.json", {"horizons": horizon_audits, "candidates": records})
            del cache
    selected = {family: max((r for r in records if r["family"] == family), key=rank)
                for family in ("mlp", "logistic", "tree")}
    selected["fixed5_mlp"] = max((r for r in records if r["family"] == "mlp" and r["horizon_seconds"] == 5), key=rank)
    selected["legacy_mlp"] = next(r for r in records if r["base_name"] == "legacy8" and
                                  r["horizon_seconds"] == selected["mlp"]["horizon_seconds"])
    selection = {"selected": selected, "horizons": horizon_audits, "candidates": records}
    dump(output / "selection.json", selection)
    joblib.dump({key: bundles[record["name"]] for key, record in selected.items()}, output / "frozen.joblib")
    dump(output / "frozen-manifest.json", {name: sha256_file(output / name)
                                          for name in ("protocol.json", "selection.json", "frozen.joblib")})
    print("Selection frozen. Final test has not been scored.", flush=True)
    return selection


def evaluate(source, output):
    if (output / "report.json").exists():
        raise ValueError("This final test was already evaluated; do not retune against it")
    manifest = json.loads((output / "frozen-manifest.json").read_text())
    for name in ("protocol.json", "selection.json", "frozen.joblib"):
        if manifest.get(name) != sha256_file(output / name):
            raise ValueError("Frozen development artifact changed")
    protocol = json.loads((output / "protocol.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    rows, metadata = load_source(source)
    if metadata["sha256"] != protocol["source_sha256"] or metadata["split_days"] != protocol["split_days"]:
        raise ValueError("Evaluation source differs from frozen development protocol")
    # Only load a locally generated, hash-checked training artifact.
    bundles = joblib.load(output / "frozen.joblib")
    metrics, comparisons = {}, {}
    with threadpool_limits(limits=1):
        for name, bundle in bundles.items():
            split = split_for(rows, bundle["config"], metadata["split_days"], "test")
            x, y = scaled_split(split, bundle["scaler"])
            pred = predictions(bundle, x)
            metrics[name] = classification_metrics(y, pred)
            baseline = [1] * len(y)
            audit = next(a for a in selection["horizons"] if a["horizon_seconds"] == bundle["config"].horizon)
            baseline_predictions = {"always_stationary": baseline,
                                    "majority_train_class": [audit["majority_train_class_id"]] * len(y),
                                    "last_move_persistence": [movement_label(rows[e["decision_index"] - 1].mid,
                                        rows[e["decision_index"]].mid, bundle["config"]) for e in split["examples"]]}
            best = baseline_predictions[audit["best_baseline"]]
            comparisons[name] = {"horizon_seconds": bundle["config"].horizon,
                                 "always_stationary": classification_metrics(y, baseline),
                                 "development_selected_baseline_name": audit["best_baseline"],
                                 "development_selected_baseline": classification_metrics(y, best),
                                 "paired_accuracy_vs_development_baseline": paired_accuracy_interval(y, pred, best,
                                      [e["timestamp_ns"] for e in split["examples"]], seed=protocol["seed"]),
                                 "paired_accuracy": paired_accuracy_interval(y, pred, baseline,
                                      [e["timestamp_ns"] for e in split["examples"]], seed=protocol["seed"])}
        bundle = bundles["mlp"]
        config, scaler, model = bundle["config"], bundle["scaler"], bundle["model"]
        first, second, first_scales, second_scales = bundle["quantized"]
        splits = {name: split_for(rows, config, metadata["split_days"], name) for name in ("train", "validation", "test")}
        x_test, y_test = scaled_split(splits["test"], scaler)
        metrics["int8_mlp"] = metrics.pop("mlp")
        metrics["float_mlp"] = classification_metrics(y_test, model.predict(x_test).tolist())
        metrics["always_stationary"] = comparisons["mlp"]["always_stationary"]
        y_train = [e["label"] for e in splits["train"]["examples"]]
        majority = max(range(3), key=y_train.count)
        metrics["majority_train_class"] = classification_metrics(y_test, [majority] * len(y_test))
        momentum = [movement_label(rows[e["decision_index"] - 1].mid, rows[e["decision_index"]].mid, config)
                    for e in splits["test"]["examples"]]
        metrics["last_move_persistence"] = classification_metrics(y_test, momentum)
        inputs, hidden_acc, hidden, output_acc, logits = mlp_batch(first, second, x_test, first_scales["input_scale_exponent"])
    vectors = []
    for index, example in enumerate(splits["test"]["examples"]):
        x = inputs[index].tolist()
        ha, h = first.evaluate(x)
        oa, scores = second.evaluate(h)
        if (ha, h, oa, scores) != (hidden_acc[index].tolist(), hidden[index].tolist(), output_acc[index].tolist(), logits[index].tolist()):
            raise AssertionError("Batched integer arithmetic differs from scalar reference")
        vectors.append({"dataset_index": example["decision_index"], "timestamp_ns": example["timestamp_ns"],
                        "label": example["label"], "inputs": x, "hidden_accumulators": ha, "hidden": h,
                        "output_accumulators": oa, "outputs": scores})
    export_model(first, output / "layer0")
    export_model(second, output / "layer1")
    network = {"schema_version": 1, "task": "lob_direction", "dataset_kind": "real",
               "evaluation_protocol": "calendar_days_v1", "split_days": metadata["split_days"],
               "architecture": [first.inputs, first.outputs, second.outputs], "classes": CLASSES,
               "seed": protocol["seed"], "feature_config": asdict(config), "feature_names": feature_names(config),
               "layers": [{"directory": "layer0", **first_scales}, {"directory": "layer1", **second_scales}]}
    split_metadata = {name: {"raw_start": split["raw_start"], "raw_stop": split["raw_stop"],
                            "indices": [{k: e[k] for k in ("feature_start", "decision_index", "label_end")} for e in split["examples"]]}
                      for name, split in splits.items()}
    files = {"network.json": network, "preprocessing.json": asdict(scaler), "split.json": split_metadata,
             "vectors.json": {"schema_version": 1, "dataset_kind": "real", "vectors": vectors},
             "float_parameters.json": {"weights": [w.T.tolist() for w in model.coefs_], "biases": [b.tolist() for b in model.intercepts_]}}
    for name, content in files.items():
        dump(output / name, content)
    diagnostics = {name: input_diagnostics([e["features"] for e in split["examples"]], scaler,
                                           first_scales["input_scale_exponent"], feature_names(config))
                   for name, split in splits.items()}
    artifact_names = [*files, *manifest, "frozen-manifest.json"] + [
        f"layer{i}/{name}" for i in (0, 1) for name in ("model.json", "weights.mem", "biases.mem")]
    report = {"schema_version": 1, "task": "lob_direction", "dataset_kind": "real",
              "evaluation_protocol": "calendar_days_v1", "source_sha256": metadata["sha256"],
              "source_metadata": metadata, "architecture": network["architecture"], "seed": protocol["seed"],
              "selection": selection, "samples": {name: len(s["examples"]) for name, s in splits.items()},
              "normalization_and_calibration_split": "train days only; no validation refit",
              "test": metrics, "comparisons": comparisons, "input_diagnostics": diagnostics,
              "all_batched_test_intermediates_match_scalar": True,
              "accuracy_drop_percentage_points": 100 * (metrics["float_mlp"]["accuracy"] - metrics["int8_mlp"]["accuracy"]),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "sha256": {name: sha256_file(output / name) for name in artifact_names}}
    dump(output / "report.json", report)
    print(json.dumps({"architecture": network["architecture"], "test": metrics, "comparisons": comparisons}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/nasdaq-days/snapshots.csv"))
    parser.add_argument("--output", type=Path, default=Path("build/prediction-quality"))
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--phase", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--synthetic-model", type=Path)
    args = parser.parse_args()
    if args.phase == "develop":
        develop(args.input, args.output, args.seed, args.synthetic_model)
    else:
        evaluate(args.input, args.output)


if __name__ == "__main__":
    main()
