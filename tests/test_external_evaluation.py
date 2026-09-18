import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_external_lob import evaluate
from scripts.fetch_itch_sample import sha256_file
from scripts.generate_lob import generate
from tests.export_fixture import create_export


class ExternalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training, _ = create_export(self.root)
        self.source = self.root / "external.csv"
        generate(self.source, rows=120, levels=1, seed=404)
        self.sidecar = self.source.with_suffix(".metadata.json")
        self.metadata = json.loads(self.sidecar.read_text())
        # Only a unit fixture for the real-source metadata gate, not a market dataset.
        self.metadata["dataset_kind"] = "real"
        self.metadata["source"] = "unit fixture"
        self.sidecar.write_text(json.dumps(self.metadata))

    def test_scores_all_supported_windows_without_modifying_the_model(self):
        before = {p.name: sha256_file(p) for p in self.root.glob("*.json")}
        result = evaluate(self.root, self.source, self.root / "evaluation.json")
        self.assertEqual(result["examples"], 114)  # 120 - lookback 2 - horizon 5 + 1
        self.assertEqual(sum(result["metrics"]["support"].values()), 114)
        self.assertFalse(result["rtl_verified"])
        self.assertEqual(result["model_training_dataset_kind"], "synthetic")
        for name, digest in before.items():
            self.assertEqual(sha256_file(self.root / name), digest)

    def test_reject_source_checksum_or_configuration_mismatch(self):
        for key, value in (("sha256", "bad"), ("levels", 5), ("tick_size", 1.0), ("dataset_kind", "synthetic")):
            self.sidecar.write_text(json.dumps({**self.metadata, key: value}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "metadata mismatch"):
                evaluate(self.root, self.source, self.root / "evaluation.json")

    def test_reject_training_source_as_external(self):
        with self.assertRaisesRegex(ValueError, "training source"):
            evaluate(self.root, self.training, self.root / "evaluation.json")
