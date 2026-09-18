"""Small complete export fixture; no training or third-party packages required."""

from dataclasses import asdict
import hashlib
import json

from models.lob.data import CLASSES, FeatureConfig, Standardizer, feature_names, prepare, read_snapshots
from models.lob.metrics import classification_metrics
from models.quantization.calibration import quantize_value
from models.quantization.dense_reference import DenseSpec, export_model
from scripts.generate_lob import generate


def rehash(path):
    report = json.loads((path / "report.json").read_text())
    report["sha256"] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                        for name in report["sha256"]}
    (path / "report.json").write_text(json.dumps(report))


def create_export(path):
    config = FeatureConfig(levels=1, lookback=2)
    source = path / "snapshots.csv"
    generate(source, rows=120, levels=1)
    rows = read_snapshots(source, 1)
    splits = prepare(rows, config)
    scaler = Standardizer.fit([e["features"] for e in splits["train"]["examples"]])
    first = DenseSpec(((1,) * 12, (-1,) * 12), (0, 0), shift=2, relu=True)
    second = DenseSpec(((1, -1), (0, 0), (-1, 1)), (0, 1, 0), shift=0)
    export_model(first, path / "layer0")
    export_model(second, path / "layer1")
    network = {"schema_version": 1, "task": "lob_direction", "classes": CLASSES,
               "dataset_kind": "synthetic", "architecture": [12, 2, 3],
               "feature_config": asdict(config), "feature_names": feature_names(config),
               "layers": [{"directory": "layer0", "input_scale_exponent": 0, "weight_scale_exponent": 0,
                           "bias_scale_exponent": 0, "output_scale_exponent": 2},
                          {"directory": "layer1", "input_scale_exponent": 2, "weight_scale_exponent": 0,
                           "bias_scale_exponent": 2, "output_scale_exponent": 2}]}
    vectors = []
    for e in splits["test"]["examples"]:
        inputs = [quantize_value(v, 0) for v in scaler.transform(e["features"])]
        hidden_acc, hidden = first.evaluate(inputs)
        output_acc, outputs = second.evaluate(hidden)
        vectors.append({"dataset_index": e["decision_index"], "timestamp_ns": e["timestamp_ns"],
                        "label": e["label"], "inputs": inputs, "hidden_accumulators": hidden_acc,
                        "hidden": hidden, "output_accumulators": output_acc, "outputs": outputs})
    split_data = {name: {"raw_start": split["raw_start"], "raw_stop": split["raw_stop"],
                        "indices": [{key: e[key] for key in ("feature_start", "decision_index", "label_end")}
                                    for e in split["examples"]]} for name, split in splits.items()}
    files = {"network.json": network, "preprocessing.json": asdict(scaler), "split.json": split_data,
             "vectors.json": {"schema_version": 1, "vectors": vectors}, "float_parameters.json": {}}
    for name, data in files.items():
        (path / name).write_text(json.dumps(data))
    names = list(files) + [f"layer{i}/{name}" for i in (0, 1)
                           for name in ("model.json", "weights.mem", "biases.mem")]
    metrics = classification_metrics([v["label"] for v in vectors],
                                     [max(range(3), key=lambda i: v["outputs"][i]) for v in vectors])
    report = {"schema_version": 1, "dataset_kind": "synthetic", "architecture": [12, 2, 3],
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "samples": {name: len(split["examples"]) for name, split in splits.items()},
              "test": {"int8_mlp": metrics}, "sha256": {name: "" for name in names}}
    (path / "report.json").write_text(json.dumps(report))
    rehash(path)
    return source, rows
