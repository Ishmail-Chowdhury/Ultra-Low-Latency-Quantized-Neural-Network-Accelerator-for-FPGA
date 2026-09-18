"""Load an exported LOB model and classify a causal snapshot window in software."""

import hashlib
import json
from pathlib import Path

from models.lob.data import CLASSES, FeatureConfig, Standardizer, feature_window
from models.quantization.calibration import quantize_value
from models.quantization.dense_reference import DenseSpec


class LobPredictor:
    def __init__(self, directory: Path):
        report = json.loads((directory / "report.json").read_text())
        for name, expected in report["sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Model artifact checksum mismatch: {name}")
        self.network = json.loads((directory / "network.json").read_text())
        if self.network.get("task") != "lob_direction" or self.network.get("classes") != CLASSES:
            raise ValueError("not a compatible LOB direction model")
        self.config = FeatureConfig(**self.network["feature_config"])
        scaling = json.loads((directory / "preprocessing.json").read_text())
        self.scaler = Standardizer(tuple(scaling["mean"]), tuple(scaling["scale"]))
        self.input_exponent = self.network["layers"][0]["input_scale_exponent"]
        self.first = DenseSpec.from_dict(json.loads((directory / "layer0/model.json").read_text()))
        self.second = DenseSpec.from_dict(json.loads((directory / "layer1/model.json").read_text()))
        if self.first.outputs != self.second.inputs or self.second.outputs != 3:
            raise ValueError("invalid LOB network dimensions")

    def predict(self, rows, decision: int) -> dict:
        values = self.scaler.transform(feature_window(rows, decision, self.config))
        inputs = [quantize_value(value, self.input_exponent) for value in values]
        _, hidden = self.first.evaluate(inputs)
        _, logits = self.second.evaluate(hidden)
        label = max(range(3), key=lambda i: logits[i])
        return {"decision_index": decision, "timestamp_ns": rows[decision].timestamp_ns,
                "class_id": label, "prediction": CLASSES[label], "logits": logits}
