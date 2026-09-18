import json
from pathlib import Path
import tempfile
import unittest

from models.lob.inference import LobPredictor
from models.quantization.artifacts import validate_export
from scripts.verify_lob_replay import verify
from tests.export_fixture import create_export, rehash


class ExportValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.source, _ = create_export(self.path)

    def edit(self, name, mutate):
        path = self.path / name
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))
        rehash(self.path)

    def test_valid_complete_export(self):
        validate_export(self.path)
        report = verify(self.path, self.source)
        self.assertEqual(report["samples"], 18)
        self.assertTrue(report["labels_and_splits_verified"])

    def test_manifest_cannot_omit_artifacts(self):
        self.edit("report.json", lambda r: r["sha256"].pop("layer0/weights.mem"))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_export(self.path)

    def test_wrong_memory_detected_even_when_checksum_matches(self):
        (self.path / "layer0/weights.mem").write_text("00\n" * 24)
        rehash(self.path)
        with self.assertRaisesRegex(ValueError, "Weight memory"):
            validate_export(self.path)

    def test_truncated_bias_memory(self):
        (self.path / "layer1/biases.mem").write_text("00000000\n")
        rehash(self.path)
        with self.assertRaisesRegex(ValueError, "Bias memory"):
            validate_export(self.path)

    def test_inconsistent_scale_metadata(self):
        self.edit("network.json", lambda n: n["layers"][0].update(output_scale_exponent=3))
        with self.assertRaisesRegex(ValueError, "Scale metadata"):
            validate_export(self.path)

    def test_feature_order_is_validated(self):
        self.edit("network.json", lambda n: n["feature_names"].reverse())
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            LobPredictor(self.path)

    def test_wrong_label_is_detected(self):
        self.edit("vectors.json", lambda d: d["vectors"][0].update(label=(d["vectors"][0]["label"] + 1) % 3))
        with self.assertRaisesRegex(ValueError, "label"):
            verify(self.path, self.source)

    def test_missing_test_example_is_detected(self):
        self.edit("vectors.json", lambda d: d["vectors"].pop())
        with self.assertRaisesRegex(ValueError, "coverage"):
            verify(self.path, self.source)

    def test_training_normalization_is_recomputed(self):
        self.edit("preprocessing.json", lambda d: d["mean"].__setitem__(0, d["mean"][0] + 1))
        with self.assertRaisesRegex(ValueError, "training-only"):
            verify(self.path, self.source)

    def test_split_boundary_corruption_is_detected(self):
        self.edit("split.json", lambda d: d["test"].update(raw_start=1))
        with self.assertRaisesRegex(ValueError, "chronological"):
            verify(self.path, self.source)

    def test_intermediate_corruption_is_detected(self):
        self.edit("vectors.json", lambda d: d["vectors"][0]["hidden_accumulators"].__setitem__(0, 99999))
        with self.assertRaisesRegex(ValueError, "intermediate"):
            verify(self.path, self.source)

    def test_input_corruption_is_detected_even_before_final_scores(self):
        self.edit("vectors.json", lambda d: d["vectors"][0]["inputs"].__setitem__(0, 127))
        with self.assertRaisesRegex(ValueError, "Input/intermediate"):
            verify(self.path, self.source)

    def test_reported_accuracy_is_recomputed(self):
        self.edit("report.json", lambda d: d["test"]["int8_mlp"].update(accuracy=-1))
        with self.assertRaisesRegex(ValueError, "metrics"):
            verify(self.path, self.source)
