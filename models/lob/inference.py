"""Load an exported LOB model and classify a causal snapshot window in software."""

import json
from pathlib import Path

from models.lob.data import CLASSES, FeatureConfig, Standardizer, feature_names, feature_window
from models.quantization.artifacts import validate_export
from models.quantization.calibration import quantize_value


class LobPredictor:
    def __init__(self, directory: Path):
        _, self.network, self.first, self.second = validate_export(directory)
        if self.network.get("task") != "lob_direction" or self.network.get("classes") != CLASSES:
            raise ValueError("not a compatible LOB direction model")
        self.config = FeatureConfig(**self.network["feature_config"])
        scaling = json.loads((directory / "preprocessing.json").read_text())
        self.scaler = Standardizer(tuple(scaling["mean"]), tuple(scaling["scale"]))
        self.input_exponent = self.network["layers"][0]["input_scale_exponent"]
        if (self.network.get("feature_names") != feature_names(self.config) or
                len(self.scaler.mean) != self.first.inputs or
                self.first.inputs != len(feature_names(self.config))):
            raise ValueError("LOB preprocessing does not match network inputs")
        self.scaler.transform([0.0] * self.first.inputs)

    def predict(self, rows, decision: int) -> dict:
        values = self.scaler.transform(feature_window(rows, decision, self.config))
        inputs = [quantize_value(value, self.input_exponent) for value in values]
        _, hidden = self.first.evaluate(inputs)
        _, logits = self.second.evaluate(hidden)
        label = max(range(3), key=lambda i: logits[i])
        return {"decision_index": decision, "timestamp_ns": rows[decision].timestamp_ns,
                "class_id": label, "prediction": CLASSES[label], "logits": logits}
