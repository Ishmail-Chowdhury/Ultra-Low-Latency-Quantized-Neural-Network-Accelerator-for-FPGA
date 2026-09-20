from dataclasses import replace
import unittest

from models.lob.data import FeatureConfig, Snapshot, Standardizer, feature_names, feature_window


class QualityFeatureTests(unittest.TestCase):
    def test_robust_scaling_constant_columns_and_export_roundtrip(self):
        from dataclasses import asdict
        scaler = Standardizer.fit([[0, 7], [1, 7], [2, 7], [3, 7], [100, 7]], "robust", 4.)
        self.assertEqual(scaler.mean, (2., 7.))
        self.assertAlmostEqual(scaler.scale[0], 2 / 1.3489795003921634)
        self.assertEqual(scaler.transform([100, 7]), [4., 0.])
        self.assertEqual(Standardizer.from_dict(asdict(scaler)), scaler)
        self.assertEqual(Standardizer.from_dict({"mean": [1], "scale": [2]}).transform([5]), [2.])
        for bad in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Standardizer.fit([[1], [2]], clip=bad)

    def test_queue_features_flow_sign_and_no_future_access(self):
        before = Snapshot(1, "day", "AAPL", (100.,), (100.,), (100.02,), (100.,))
        after = replace(before, timestamp_ns=2, bid_sizes=(300.,))
        config = FeatureConfig(levels=1, lookback=2, include_queue_features=True)
        values = feature_window([before, after], 1, config)
        self.assertEqual(len(values), len(feature_names(config)))
        self.assertAlmostEqual(values[-3], .5)  # best bid/ask imbalance
        self.assertAlmostEqual(values[-2], .5)  # microprice offset, in ticks
        self.assertAlmostEqual(values[-1], 200 / 300)  # normalized sampled flow
        future = replace(after, timestamp_ns=3, bid_sizes=(1e9,))
        self.assertEqual(values, feature_window([before, after, future], 1, config))
        with self.assertRaises(ValueError):
            feature_window([before, replace(after, session_id="other")], 1, config)

    def test_bounded_scaler_training_statistics_ignore_future(self):
        training = [[1., 7.], [2., 7.], [3., 7.]]
        scaler = Standardizer.fit(training, "robust", 4.)
        self.assertEqual(scaler.transform([1e9, -1e9]), [4., -4.])
        self.assertEqual(scaler, Standardizer.fit(training, "robust", 4.))


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional training dependency NumPy is not installed")
class QualityDiagnosticTests(unittest.TestCase):
    def test_preprocessing_clipping_is_not_hidden_as_quantization_fix(self):
        from models.training.diagnostics import input_diagnostics
        raw = [[0., 0.], [100., -100.]]
        unbounded = input_diagnostics(raw, Standardizer((0., 0.), (1., 1.)), -4, ["a", "b"])
        bounded = input_diagnostics(raw, Standardizer((0., 0.), (1., 1.), clip=4.), -4, ["a", "b"])
        self.assertEqual(unbounded["int8_saturation_fraction"], .5)
        self.assertEqual(bounded["int8_saturation_fraction"], 0.)
        self.assertEqual(bounded["preprocessing_clip_fraction"], .5)

    def test_block_bootstrap_keeps_comparison_paired(self):
        from models.training.diagnostics import paired_accuracy_interval
        y = [0, 1, 2] * 100
        times = [i * 10**9 for i in range(len(y))]
        same = paired_accuracy_interval(y, y, y, times, repetitions=100)
        self.assertEqual(same["interval_95_percentage_points"], [0., 0.])
        self.assertEqual(same["blocks"], 5)
        improved = paired_accuracy_interval(y, y, [(v + 1) % 3 for v in y], times, repetitions=100)
        self.assertEqual(improved["interval_95_percentage_points"], [100., 100.])


try:
    import sklearn
except ImportError:
    sklearn = None


@unittest.skipIf(sklearn is None, "optional training dependencies are not installed")
class QualityExperimentTests(unittest.TestCase):
    def test_development_never_opens_test_examples_and_frozen_export_replays(self):
        import csv
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from models.training import train_days
        from scripts.generate_lob import generate
        from scripts.fetch_itch_sample import sha256_file
        from scripts.verify_lob_replay import verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.csv"
            generate(source, rows=1200, seed=2030)
            with source.open(newline="") as stream:
                csv_rows = list(csv.reader(stream))
            days = [f"2019-07-0{i}" for i in range(1, 5)]
            for i, row in enumerate(csv_rows[1:]):
                row[0] = str(i * 10**9)
                row[1] = days[i // 300] + "/1"
            with source.open("w", newline="") as stream:
                csv.writer(stream).writerows(csv_rows)
            # Artificial fixture marked with the real-data schema solely to
            # exercise validation; never exported as a market-data result.
            metadata = {"dataset_kind": "real", "source": "artificial unit-test fixture",
                        "sha256": sha256_file(source), "interval_ns": 10**9, "tick_size": .01,
                        "levels": 5, "split_days": {"train": days[:2], "validation": days[2:3], "test": days[3:]}}
            source.with_suffix(".metadata.json").write_text(json.dumps(metadata))
            candidates = [{**train_days.CANDIDATES[0], "scaling": "robust", "clip": 4., "features": True},
                          *[c for c in train_days.CANDIDATES if c["name"] in ("logistic_bFalse", "tree_bFalse")]]
            real_split = train_days.split_for
            def development_split(rows, config, plan, name):
                self.assertNotEqual(name, "test", "development accessed final test examples")
                return real_split(rows, config, plan, name)
            output = root / "model"
            with patch.object(train_days, "CANDIDATES", candidates), patch.object(train_days, "HORIZONS", (5,)), \
                    patch.object(train_days, "split_for", side_effect=development_split):
                train_days.develop(source, output)
            selection = (output / "selection.json").read_bytes()
            train_days.evaluate(source, output)
            self.assertEqual(selection, (output / "selection.json").read_bytes())
            report = verify(output, source)
            self.assertTrue(report["training_only_preprocessing_verified"])
            self.assertTrue(report["all_intermediates_verified"])
            with self.assertRaisesRegex(ValueError, "already evaluated"):
                train_days.evaluate(source, output)
            with self.assertRaisesRegex(ValueError, "overwritten"):
                train_days.develop(source, output)
