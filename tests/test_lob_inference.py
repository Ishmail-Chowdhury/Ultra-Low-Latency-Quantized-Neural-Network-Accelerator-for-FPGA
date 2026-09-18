from pathlib import Path
import tempfile
import unittest

from models.lob.inference import LobPredictor
from models.lob.metrics import classification_metrics
from scripts.predict_lob import replay
from tests.export_fixture import create_export


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
            source, rows = create_export(path)
            model = LobPredictor(path)
            self.assertEqual(model.predict(rows, 10), model.predict(rows[:11], 10))
            self.assertEqual(model.predict(rows, 119)["timestamp_ns"], rows[-1].timestamp_ns)
            self.assertEqual(replay(path, source, path / "predictions.csv"), 119)
            with self.assertRaises(ValueError):
                model.predict(rows, 0)

    def test_model_checksums_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            create_export(path)
            (path / "layer0/weights.mem").write_text("00\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                LobPredictor(path)
