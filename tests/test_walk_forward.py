from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from models.lob.data import FeatureConfig, Standardizer, read_snapshots
from models.lob.walk_forward import balanced_weights, final_splits, plan, select_candidate, selection_folds
from scripts.generate_lob import generate


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        source = Path(self.directory.name) / "snapshots.csv"
        generate(source, rows=300, seed=2027)
        self.rows = read_snapshots(source)
        self.config = FeatureConfig()

    def test_expanding_folds_and_disjoint_support(self):
        folds = selection_folds(self.rows, self.config)
        self.assertEqual([f["train"]["raw_stop"] for f in folds], [120, 160, 200])
        for fold in folds:
            train, validation = fold["train"]["examples"], fold["validation"]["examples"]
            self.assertLess(max(e["label_end"] for e in train), min(e["feature_start"] for e in validation))
            self.assertLess(max(e["label_end"] for e in validation), plan(300)["holdout_start"])
        final = final_splits(self.rows, self.config)
        self.assertEqual(final["train"]["raw_stop"], 240)
        self.assertEqual(final["test"]["raw_start"], 240)

    def test_final_test_cannot_change_any_selection_fold(self):
        altered = [row if i < 240 else replace(row, bid_sizes=(99999,) * 5)
                   for i, row in enumerate(self.rows)]
        self.assertEqual(selection_folds(self.rows, self.config), selection_folds(altered, self.config))

    def test_validation_changes_do_not_change_fold_training_scaler(self):
        altered = [row if i < 120 else replace(row, ask_sizes=(99999,) * 5)
                   for i, row in enumerate(self.rows)]
        before = selection_folds(self.rows, self.config)[0]
        after = selection_folds(altered, self.config)[0]
        scaler = lambda fold: Standardizer.fit([e["features"] for e in fold["train"]["examples"]])
        self.assertEqual(scaler(before), scaler(after))

    def test_balancing_uses_equal_class_mass(self):
        labels = [0, 0, 0, 1, 2, 2]
        weights = balanced_weights(labels)
        self.assertAlmostEqual(sum(weights), len(labels))
        for label in range(3):
            self.assertAlmostEqual(sum(w for y, w in zip(labels, weights) if y == label), 2)
        with self.assertRaises(ValueError):
            balanced_weights([0, 1])

    def test_selection_uses_validation_scores_and_stable_ties(self):
        records = [{"name": "small", "family": "mlp", "fold_metrics": [{"macro_f1": 0.5}] * 3,
                    "test_score": 0.1},
                   {"name": "large", "family": "mlp", "fold_metrics": [{"macro_f1": 0.4}] * 3,
                    "test_score": 1.0}]
        self.assertEqual(select_candidate(records, "mlp")["name"], "small")
        records[1]["fold_metrics"] = [{"macro_f1": 0.5}] * 3
        self.assertEqual(select_candidate(records, "mlp")["name"], "small")

    def test_session_boundaries_and_short_files(self):
        rows = [row if i < 145 else replace(row, session_id="second") for i, row in enumerate(self.rows)]
        for fold in selection_folds(rows, self.config):
            for split in fold.values():
                for example in split["examples"]:
                    span = rows[example["feature_start"]:example["label_end"] + 1]
                    self.assertEqual(len({r.session_id for r in span}), 1)
        with self.assertRaises(ValueError):
            selection_folds(self.rows[:15], self.config)
