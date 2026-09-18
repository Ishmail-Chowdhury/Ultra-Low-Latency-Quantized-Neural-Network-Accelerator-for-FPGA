import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from models.lob.data import (FeatureConfig, Standardizer, feature_names, feature_window,
                             movement_label, prepare, read_snapshots)
from scripts.generate_lob import generate
from scripts.prepare_lob import build


class LobDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "book.csv"
        generate(self.path, rows=120)
        self.rows = read_snapshots(self.path)
        self.config = FeatureConfig()

    def test_generation_is_reproducible(self):
        other = self.path.with_name("other.csv")
        generate(other, rows=120)
        self.assertEqual(self.path.read_bytes(), other.read_bytes())
        generate(other, rows=120, seed=999)
        self.assertNotEqual(self.path.read_bytes(), other.read_bytes())

    def test_features_are_causal_and_have_named_order(self):
        vector = feature_window(self.rows, 10, self.config)
        self.assertEqual(len(vector), 88)
        self.assertEqual(len(feature_names(self.config)), 88)
        self.assertEqual(vector, feature_window(self.rows[:11], 10, self.config))
        row = self.rows[10]
        self.assertAlmostEqual(vector[-1], (sum(row.bid_sizes) - sum(row.ask_sizes)) /
                               (sum(row.bid_sizes) + sum(row.ask_sizes)))

    def test_labels_and_threshold_ties(self):
        for future, label in [(100.02, 2), (99.98, 0), (100, 1), (100.005, 1), (99.995, 1)]:
            with self.subTest(future=future):
                self.assertEqual(movement_label(100, future, self.config), label)

    def test_no_feature_or_label_overlap_between_splits(self):
        splits = prepare(self.rows, self.config)
        support = []
        for split in splits.values():
            used = set()
            for example in split["examples"]:
                self.assertGreaterEqual(example["feature_start"], split["raw_start"])
                self.assertLess(example["label_end"], split["raw_stop"])
                used.update(range(example["feature_start"], example["label_end"] + 1))
            support.append(used)
        for left, right in zip(support, support[1:]):
            self.assertLess(max(left), min(right))
        self.assertEqual([len(s["examples"]) for s in splits.values()], [64, 16, 16])

    def test_future_changes_cannot_change_training_scaler(self):
        original = prepare(self.rows, self.config)
        changed_rows = [row if i < 72 else replace(row, bid_sizes=(1000000,) * 5)
                        for i, row in enumerate(self.rows)]
        changed = prepare(changed_rows, self.config)
        self.assertEqual(original["train"], changed["train"])
        fit = lambda data: Standardizer.fit([e["features"] for e in data["train"]["examples"]])
        self.assertEqual(fit(original), fit(changed))
        scaler = Standardizer.fit([[1, 7], [3, 7]])
        self.assertEqual(scaler.transform([2, 7]), [0, 0])
        self.assertEqual(scaler.transform([4, 8]), [2, 1])

    def test_no_cross_session_windows_or_labels(self):
        rows = [row if i < 30 else replace(row, session_id="next-session")
                for i, row in enumerate(self.rows)]
        splits = prepare(rows, self.config)
        for split in splits.values():
            for example in split["examples"]:
                support = rows[example["feature_start"]:example["label_end"] + 1]
                self.assertEqual(len({row.session_id for row in support}), 1)
        with self.assertRaises(ValueError):
            feature_window(rows, 31, self.config)

    def test_csv_rejects_bad_snapshots(self):
        with self.path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        header = rows[0]
        mutations = [(1, "bid_size_1", "nan"), (1, "ask_size_1", "0"),
                     (1, "bid_price_1", "999"), (1, "bid_price_2", "999"),
                     (2, "timestamp_ns", "0"), (2, "symbol", "OTHER"),
                     (1, "ask_price_1", ""), (2, "session_id", "")]
        for index, name, value in mutations:
            altered = [list(row) for row in rows]
            altered[index][header.index(name)] = value
            invalid = self.path.with_name("invalid.csv")
            with invalid.open("w", newline="") as stream:
                csv.writer(stream).writerows(altered)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                read_snapshots(invalid)

    def test_export_provenance_and_metadata_integrity(self):
        output = self.path.parent / "prepared"
        summary = build(self.path, output, self.config, "synthetic", "test generator")
        document = json.loads((output / "dataset.json").read_text())
        self.assertEqual(summary["features"], 88)
        self.assertEqual(document["dataset_kind"], "synthetic")
        with self.assertRaises(ValueError):
            build(self.path, output, self.config, "real", "wrong kind")
        with self.assertRaises(ValueError):
            build(self.path, output, replace(self.config, tick_size=0.02), "synthetic", "wrong tick")
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaises(ValueError):
            build(self.path, output, self.config, "synthetic", "changed data")

    def test_invalid_configuration_or_insufficient_data(self):
        for kwargs in [dict(horizon=0), dict(lookback=0), dict(levels=0),
                       dict(tick_size=0), dict(threshold_ticks=-1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FeatureConfig(**kwargs)
        with self.assertRaises(ValueError):
            prepare(self.rows[:10], self.config)
