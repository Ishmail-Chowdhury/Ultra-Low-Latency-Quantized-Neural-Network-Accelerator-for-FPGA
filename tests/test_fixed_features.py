import csv
from dataclasses import replace
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from models.lob.data import FeatureConfig, Standardizer, feature_names, feature_window, read_snapshots
from models.lob.fixed_features import (ONE, PRICE_UNITS, QUANTITY_UNITS, FixedPreprocessor, FixedSnapshot,
    constants, decimal_units, divide_round, export_frontend, read_fixed_snapshots, shift_round)
from models.lob.inference import LobPredictor
from scripts.generate_lob import generate
from scripts.predict_lob import replay
from tests.export_fixture import create_export


class FixedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "snapshots.csv"
        generate(self.source, rows=160)
        self.rows = read_snapshots(self.source)
        self.fixed_rows = read_fixed_snapshots(self.source)

    def processor(self, config, method="standard", clip=None):
        features = [feature_window(self.rows, i, config) for i in range(config.lookback - 1, 100)]
        scaler = Standardizer.fit(features, method, clip)
        # Synthetic float price subtraction creates near-constant columns with
        # tiny nonzero variance. Use explicitly representable fixture scales;
        # unsupported exported scales are separately required to fail below.
        scaler = replace(scaler, scale=tuple(max(s, 1e-3) for s in scaler.scale))
        return FixedPreprocessor(constants(config, scaler, -4)), scaler

    def test_signed_rounding_bounds_and_saturation(self):
        for numerator, divisor, expected in ((3, 2, 2), (-3, 2, -2), (1, 2, 1), (-1, 2, -1),
                                             (1, 3, 0), (-1, 3, 0), (7, 3, 2), (-7, 3, -2)):
            self.assertEqual(divide_round(numerator, divisor), expected)
        self.assertEqual(shift_round(-(1 << 127), 1), -(1 << 126))
        for a, b in ((1, 0), (1, -1), (1 << 127, 1)):
            with self.assertRaises(ValueError):
                divide_round(a, b)
        processor, _ = self.processor(FeatureConfig())
        values = [0] * len(processor.document["centers_q24"])
        values[:4] = [1 << 19, -(1 << 19), 1000 * ONE, -1000 * ONE]
        self.assertEqual(processor.quantize(values)[:4], [1, -1, 127, -128])

    def test_exact_decimal_ingress_and_domain_rejections(self):
        self.assertEqual(decimal_units("100.0001", PRICE_UNITS), 1000001)
        self.assertEqual(decimal_units(".0001", QUANTITY_UNITS), 1)
        for value in ("0.00001", "NaN", "Infinity", "1e99999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decimal_units(value, PRICE_UNITS)
        original = self.fixed_rows[0]
        for change in ({"bid_sizes": (0,) * 5}, {"bid_sizes": (1 << 48,) * 5},
                       {"ask_prices": original.bid_prices}, {"bid_prices": (True,) * 5},
                       {"timestamp_ns": -1}, {"bid_sizes": list(original.bid_sizes)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(original, **change)
        self.assertEqual(FixedSnapshot.from_snapshot(self.rows[0]), original)
        with self.source.open() as stream:
            rows = list(csv.reader(stream))
        rows[1][3] = "100.00001"
        with self.source.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, "CSV row 2"):
            read_fixed_snapshots(self.source)

    def test_log_table_against_high_precision_reference_at_extremes(self):
        processor, _ = self.processor(FeatureConfig())
        rng = random.Random(2031)
        quantities = [0, 1, 9999, 10000, (1 << 48) - 1]
        quantities += [rng.randrange(1, 1 << rng.randrange(1, 49)) for _ in range(500)]
        quantities += [max(0, (1 << bit) - QUANTITY_UNITS + delta)
                       for bit in range(14, 49) for delta in (-1, 0, 1)]
        with localcontext() as context:
            context.prec = 60
            for quantity in quantities:
                if quantity >= 1 << 48:
                    continue
                actual = Decimal(processor.log_quantity_q24(quantity)) / ONE
                expected = (1 + Decimal(quantity) / QUANTITY_UNITS).ln()
                self.assertLess(abs(actual - expected), Decimal("0.00000025"), quantity)
        self.assertEqual(processor.log_integer_q30(1), 0)
        self.assertEqual(processor.log_integer_q30(2), processor.document["ln2_q30"])

    def test_all_feature_modes_match_float_order_and_error_budget(self):
        for changes, queues in ((False, False), (True, False), (False, True), (True, True)):
            config = FeatureConfig(include_mid_changes=changes, include_queue_features=queues)
            processor, scaler = self.processor(config, "robust", 4.)
            for index in (3, 4, 57, 99, 159):
                fixed = processor.feature_window(self.fixed_rows, index)
                floating = feature_window(self.rows, index, config)
                self.assertEqual(len(fixed), len(feature_names(config)))
                self.assertTrue(all(type(v) is int for v in fixed))
                self.assertLess(max(abs(a / ONE - b) for a, b in zip(fixed, floating)), 2e-6)
                normalized = processor.normalize(fixed)
                expected = scaler.transform(floating)
                self.assertLess(max(abs(a / ONE - b) for a, b in zip(normalized, expected)), 1e-5)
                self.assertTrue(all(abs(v) <= 4 * ONE for v in normalized))

    def test_half_tick_and_snapshot_flow_do_not_truncate_midpoint(self):
        row = FixedSnapshot(0, "s", "A", (1000000,), (1000000,), (1000100,), (1000000,))
        after = replace(row, timestamp_ns=1, bid_prices=(1000001,), bid_sizes=(3000000,))
        config = FeatureConfig(levels=1, lookback=2, include_mid_changes=True, include_queue_features=True)
        count = len(feature_names(config))
        processor = FixedPreprocessor(constants(config, Standardizer((0.,) * count, (1.,) * count), -4))
        values = processor.feature_window([row, after], 1)
        self.assertEqual(values[0], ONE // 2)
        self.assertEqual(values[12], divide_round(ONE, 200))  # midpoint moves 0.005 tick
        self.assertEqual(values[-3], ONE // 2)  # queue imbalance
        self.assertEqual(values[-1], ONE)  # bid improved: new size enters normalized flow
        swapped = replace(after, bid_sizes=(1000000,), ask_sizes=(3000000,))
        self.assertLess(processor.feature_window([row, swapped], 1)[-3], 0)

    def test_runtime_is_causal_integer_only_and_rejects_session_crossing(self):
        processor, _ = self.processor(FeatureConfig(include_mid_changes=True, include_queue_features=True))
        with patch("math.log1p", side_effect=AssertionError("float log at runtime")), \
                patch("models.lob.fixed_features.Decimal", side_effect=AssertionError("Decimal at runtime")):
            actual = processor.inputs(self.fixed_rows, 10)
            self.assertEqual(actual, processor.inputs(self.fixed_rows[:11], 10))
            self.assertTrue(all(type(v) is int for v in actual))
        bad = list(self.fixed_rows)
        bad[9] = replace(bad[9], session_id="other")
        with self.assertRaises(ValueError):
            processor.inputs(bad, 10)
        with self.assertRaises(ValueError):
            processor.inputs(self.rows, 10)
        with self.assertRaises(ValueError):
            processor.inputs(self.fixed_rows, 1)

    def test_export_load_binding_rom_corruption_and_inference(self):
        model = self.root / "model"
        model.mkdir()
        source, rows = create_export(model)
        frontend = self.root / "frontend"
        export_frontend(model, frontend)
        predictor = LobPredictor(model, frontend=frontend)
        fixed = read_fixed_snapshots(source, 1)
        self.assertEqual(predictor.predict(fixed, 10), predictor.predict(fixed[:11], 10))
        self.assertEqual(replay(model, source, self.root / "predictions.csv", frontend=frontend), 119)
        with self.assertRaises(ValueError):
            export_frontend(model, model / "frontend")
        with self.assertRaises(ValueError):
            export_frontend(model, frontend)
        original = (frontend / "frontend.json").read_text()
        changed = json.loads(original)
        changed["centers_q24"][0] += 1
        (frontend / "frontend.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "constants"):
            FixedPreprocessor.load(frontend, model)
        (frontend / "frontend.json").write_text(original)
        (frontend / "log_lut.mem").write_text("00000000\n")
        with self.assertRaisesRegex(ValueError, "ROM mismatch"):
            FixedPreprocessor.load(frontend, model)

    def test_unrepresentable_constants_and_overflow_are_rejected(self):
        config = FeatureConfig()
        count = len(feature_names(config))
        for scaler, exponent in ((Standardizer((0.,) * count, (1e-11,) * count), -4),
                                  (Standardizer((0.,) * count, (1.,) * count, clip=1e-10), -4),
                                  (Standardizer((0.,) * count, (1.,) * count), -25)):
            with self.assertRaises(ValueError):
                constants(config, scaler, exponent)
        processor, _ = self.processor(config)
        with self.assertRaises(ValueError):
            processor.normalize([1 << 63] * count)
        with self.assertRaises(ValueError):
            processor.ratio_q24(1 << 120, 1)
