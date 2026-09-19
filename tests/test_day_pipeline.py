import csv
from dataclasses import replace
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from models.lob.data import FeatureConfig, Snapshot, feature_names, feature_window
from models.lob.day_split import splits, validate_plan
from models.lob.itch import messages
from scripts.fetch_itch_days import filter_symbol
from scripts.prepare_itch_days import sample_day
from tests.test_market_imports import message


def framed(bodies, terminator=True):
    return b"".join(len(b).to_bytes(2, "big") + b for b in bodies) + (b"\x00\x00" if terminator else b"")


def complete_day():
    return [message("S", time=0), message("R", time=1), message("H", time=2),
            message("S", time=3, event="S"), message("S", time=4, event="Q"),
            message("A", time=5), message("A", time=5, ref=2, side="S", price=1000100),
            message("A", time=7, ref=3, size=50), message("D", time=8, ref=3),
            message("S", time=10, event="M"), message("S", time=11, event="E"),
            message("S", time=12, event="C")]


class DayPipelineTests(unittest.TestCase):
    def test_filter_preserves_symbol_events_and_complete_markers(self):
        bodies = complete_day()
        bodies.insert(6, message("A", time=5, locate=2))
        target = io.BytesIO()
        audit = filter_symbol(io.BytesIO(framed(bodies)), target, "AAPL")
        self.assertEqual(target.getvalue(), framed(complete_day()))
        self.assertEqual(audit["all_messages"], 13)
        self.assertTrue(audit["binaryfile_terminator_seen"])
        self.assertEqual(len(list(messages(io.BytesIO(target.getvalue())))), 12)

    def test_filter_rejects_incomplete_days_trailing_data_and_bad_frames(self):
        data = framed(complete_day())
        for bad in (data + b"x", data[:-1], framed(complete_day()[:-1]), b"\x00\x01x"):
            with self.subTest(data=bad), self.assertRaises(ValueError):
                filter_symbol(io.BytesIO(bad), io.BytesIO(), "AAPL")
        with self.assertRaisesRegex(ValueError, "terminator"):
            list(messages(io.BytesIO(data + b"x")))

    def test_regular_grid_uses_no_future_updates_and_includes_same_time_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "day.itch"
            source.write_bytes(framed(complete_day()))
            output = io.StringIO()
            audit = sample_day(source, csv.writer(output), "2019-07-30", "AAPL",
                               interval_ns=1, levels=1, open_ns=5, close_ns=10)
            rows = list(csv.reader(io.StringIO(output.getvalue())))
            self.assertEqual(audit["rows"], 5)
            # At 5 and 6: size 100. Update at 7 raises to 150, delete at 8 restores 100.
            self.assertEqual([int(row[4]) for row in rows], [100, 100, 150, 100, 100])

    def test_day_partition_and_session_windows_never_cross(self):
        rows = []
        for d in range(1, 5):
            for i in range(20):
                rows.append(Snapshot(d * 100 + i, f"2019-07-0{d}/1", "AAPL",
                                     (100.,), (10.,), (100.01,), (20.,)))
        plan = {"train": ["2019-07-01", "2019-07-02"], "validation": ["2019-07-03"], "test": ["2019-07-04"]}
        config = FeatureConfig(levels=1)
        result = splits(rows, config, plan)
        self.assertEqual([len(result[n]["examples"]) for n in ("train", "validation", "test")], [24, 12, 12])
        for part in result.values():
            for e in part["examples"]:
                self.assertEqual(rows[e["feature_start"]].session_id, rows[e["label_end"]].session_id)
        changed = [replace(r, bid_sizes=(999.,)) if i >= 40 else r for i, r in enumerate(rows)]
        self.assertEqual(splits(changed, config, plan)["train"], result["train"])
        with self.assertRaises(ValueError):
            validate_plan(rows, {**plan, "test": plan["validation"]})

    def test_mid_change_features_are_causal_and_named(self):
        rows = [Snapshot(i, "day", "AAPL", (100. + i * .01,), (10.,), (100.01 + i * .01,), (20.,))
                for i in range(6)]
        config = FeatureConfig(levels=1, include_mid_changes=True)
        values = feature_window(rows, 3, config)
        self.assertEqual(len(values), len(feature_names(config)))
        for value in values[-3:]:
            self.assertAlmostEqual(value, 1.)
        self.assertEqual(values, feature_window(rows[:4], 3, config))
        with self.assertRaises(ValueError):
            FeatureConfig(include_mid_changes="yes")

    @unittest.skipUnless(shutil.which("cc"), "optional native scanner needs a C compiler")
    def test_native_filter_matches_python_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "filter"
            subprocess.run(["cc", "-O2", "-Wall", "-Wextra", "-Werror", "scripts/itch_filter.c", "-o", str(binary)], check=True, capture_output=True)
            data = framed(complete_day())
            target = io.BytesIO()
            filter_symbol(io.BytesIO(data), target, "AAPL")
            result = subprocess.run([str(binary), "AAPL"], input=data, capture_output=True, check=True)
            self.assertEqual(result.stdout, target.getvalue())
            for bad in (data + b"x", data[:-1], framed(complete_day()[:-1])):
                self.assertNotEqual(subprocess.run([str(binary), "AAPL"], input=bad, capture_output=True).returncode, 0)
