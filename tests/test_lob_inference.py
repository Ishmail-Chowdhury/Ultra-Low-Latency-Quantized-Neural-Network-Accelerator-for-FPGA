import json
from pathlib import Path
import tempfile
import unittest

from models.lob.data import CLASSES, read_snapshots
from models.lob.inference import LobPredictor
from models.lob.metrics import classification_metrics
from models.quantization.dense_reference import DenseSpec, export_model
from scripts.generate_lob import generate
from scripts.predict_lob import replay


class LobInferenceTests(unittest.TestCase):
    def test_fixed_class_metrics(self):
        metrics = classification_metrics([0, 1, 2, 2], [0, 0, 2, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [0, 1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 4 / 9)
        self.assertEqual(classification_metrics([0], [0])["macro_f1"], 1 / 3)
        with self.assertRaises(ValueError):
            classification_metrics([], [])

    def test_loaded_model_replay_and_prefix_causality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "snapshots.csv"
            generate(source, rows=20, levels=1)
            rows = read_snapshots(source, levels=1)
            first = DenseSpec(((1,) * 12, (-1,) * 12), (0, 0), shift=2, relu=True)
            second = DenseSpec(((1, -1), (0, 0), (-1, 1)), (0, 1, 0), shift=0)
            export_model(first, path / "layer0")
            export_model(second, path / "layer1")
            network = {"task": "lob_direction", "classes": CLASSES, "dataset_kind": "synthetic",
                       "feature_config": {"levels": 1, "lookback": 2, "horizon": 5},
                       "layers": [{"input_scale_exponent": 0}]}
            (path / "network.json").write_text(json.dumps(network))
            (path / "preprocessing.json").write_text(json.dumps({"mean": [0] * 12, "scale": [1] * 12}))
            (path / "report.json").write_text(json.dumps({"sha256": {}}))
            model = LobPredictor(path)
            self.assertEqual(model.predict(rows, 10), model.predict(rows[:11], 10))
            self.assertEqual(model.predict(rows, 19)["timestamp_ns"], rows[-1].timestamp_ns)
            self.assertEqual(replay(path, source, path / "predictions.csv"), 19)
            with self.assertRaises(ValueError):
                model.predict(rows, 0)

    def test_model_checksums_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "weights.mem").write_text("00\n")
            (path / "report.json").write_text(json.dumps({"sha256": {"weights.mem": "invalid"}}))
            with self.assertRaisesRegex(ValueError, "checksum"):
                LobPredictor(path)
