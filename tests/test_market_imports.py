import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from models.lob.data import read_snapshots
from models.lob.itch import ItchBook, LENGTHS, messages
from scripts.import_itch import convert as convert_itch
from scripts.import_lobster import convert as convert_lobster


def message(kind, time=1, locate=1, **fields):
    body = bytearray(LENGTHS[kind])
    body[0] = ord(kind)
    body[1:3] = locate.to_bytes(2, "big")
    body[5:11] = time.to_bytes(6, "big")
    if kind == "S":
        body[11] = ord(fields.get("event", "O"))
    elif kind == "R":
        body[11:19] = fields.get("symbol", "AAPL").ljust(8).encode()
    elif kind == "H":
        body[11:19] = b"AAPL    "
        body[19] = ord(fields.get("state", "T"))
    elif kind in ("A", "F"):
        body[11:19] = fields.get("ref", 1).to_bytes(8, "big")
        body[19] = ord(fields.get("side", "B"))
        body[20:24] = fields.get("size", 100).to_bytes(4, "big")
        body[24:32] = b"AAPL    "
        body[32:36] = fields.get("price", 1000000).to_bytes(4, "big")
    elif kind in ("E", "C", "X", "D", "U"):
        body[11:19] = fields.get("ref", 1).to_bytes(8, "big")
        if kind in ("E", "C", "X"):
            body[19:23] = fields.get("size", 20).to_bytes(4, "big")
        if kind == "U":
            body[19:27] = fields.get("new", 2).to_bytes(8, "big")
            body[27:31] = fields.get("size", 80).to_bytes(4, "big")
            body[31:35] = fields.get("price", 999900).to_bytes(4, "big")
    return bytes(body)


def initialized_book():
    book = ItchBook("AAPL")
    for kind in ("S", "R", "H"):
        book.apply(message(kind))
    return book


class ItchTests(unittest.TestCase):
    def test_order_lifecycle_and_aggregation(self):
        book = initialized_book()
        book.apply(message("A"))
        book.apply(message("F", ref=3, size=50))
        self.assertEqual(book.depth["B"][1000000], 150)
        for kind in ("E", "C", "X"):
            book.apply(message(kind))
        self.assertEqual(book.orders[1][2], 40)
        book.apply(message("U"))
        self.assertNotIn(1, book.orders)
        self.assertEqual(book.orders[2], ("B", 999900, 80))
        self.assertEqual(book.depth["B"][1000000], 50)
        book.apply(message("D", ref=3))
        self.assertNotIn(1000000, book.depth["B"])
        book.apply(message("E", ref=2, size=80))
        self.assertFalse(book.depth["B"])

    def test_prints_do_not_change_book_and_other_symbols_are_ignored(self):
        book = initialized_book()
        book.apply(message("A"))
        for kind in ("P", "Q", "B"):
            self.assertFalse(book.apply(message(kind))[1])
        book.apply(message("A", locate=2))
        self.assertEqual(book.orders, {1: ("B", 1000000, 100)})

    def test_reject_gaps_over_reductions_duplicates_and_time_reversal(self):
        for bad in (message("D"), message("E", size=101), message("A")):
            book = initialized_book()
            if bad[:1] != b"D":
                book.apply(message("A"))
            with self.subTest(kind=bad[:1]), self.assertRaises(ValueError):
                book.apply(bad)
        book = initialized_book()
        book.apply(message("A", time=3))
        with self.assertRaisesRegex(ValueError, "decrease"):
            book.apply(message("D", time=2))
        with self.assertRaisesRegex(ValueError, "Start of Messages"):
            ItchBook("AAPL").apply(message("A"))

    def test_binary_framing_rejects_truncation_and_unknown_type(self):
        body = message("S")
        data = len(body).to_bytes(2, "big") + body
        self.assertEqual(list(messages(io.BytesIO(data))), [body])
        for bad in (data[:-1], b"\x00", b"\x00\x01x", b"\x00\x0b" + b"?" * 11):
            with self.subTest(data=bad), self.assertRaises(ValueError):
                list(messages(io.BytesIO(bad)))

    def test_snapshot_conversion_breaks_sessions_at_halts_and_preserves_ties(self):
        bodies = [message(k) for k in ("S", "R", "H")]
        bodies += [message("A"), message("A", ref=2, side="S", price=1000100)]
        bodies += [message("A", ref=3), message("H", state="H"), message("H", state="T")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.itch", root / "out.csv"
            source.write_bytes(b"".join(len(b).to_bytes(2, "big") + b for b in bodies))
            meta = convert_itch(source, target, "AAPL", "2019-01-30", "unit fixture", levels=1)
            rows = read_snapshots(target, 1)
            self.assertEqual([r.timestamp_ns for r in rows], [1, 2, 3])
            self.assertEqual(rows[0].bid_prices, (100.0,))
            self.assertEqual(rows[1].bid_sizes, (200.0,))
            self.assertEqual(rows[0].session_id, rows[1].session_id)
            self.assertNotEqual(rows[1].session_id, rows[2].session_id)
            self.assertEqual(meta["timestamp_adjustments"], 2)
            old = target.read_bytes()
            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                convert_itch(source, target, "AAPL", "2019-01-30", "fixture", levels=1)
            self.assertEqual(target.read_bytes(), old)


class LobsterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.msg, self.book, self.out = [self.root / name for name in ("message.csv", "book.csv", "out.csv")]

    def write(self, messages, books):
        for path, rows in ((self.msg, messages), (self.book, books)):
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)

    def convert(self):
        return convert_lobster(self.msg, self.book, self.out, "AAPL", "2012-06-21", "unit fixture", levels=1)

    def test_scale_alignment_and_duplicate_timestamps(self):
        self.write([["34200.000000001", 1, 1, 10, 1000000, 1]] * 2,
                   [[1000100, 20, 1000000, 10], [1000200, 25, 1000000, 15]])
        meta = self.convert()
        rows = read_snapshots(self.out, 1)
        self.assertEqual(rows[0].ask_prices, (100.01,))
        self.assertEqual(rows[1].ask_sizes, (25.0,))
        self.assertEqual(rows[1].timestamp_ns - rows[0].timestamp_ns, 1)
        self.assertEqual(meta["timestamp_adjustments"], 1)

    def test_gaps_halts_and_auctions_split_sessions(self):
        messages_ = [[34200 + i, 1, 1, 10, 1000000, 1] for i in range(8)]
        messages_[2][1], messages_[2][4] = 7, -1
        messages_[4][1], messages_[4][4] = 7, 1
        messages_[6][1] = 6
        books = [[1000100, 20, 1000000, 10] for _ in messages_]
        books[1][1] = 0
        self.write(messages_, books)
        meta = self.convert()
        rows = read_snapshots(self.out, 1)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r.session_id for r in rows}), 3)
        self.assertEqual(meta["skipped_rows"], 5)

    def test_reject_misalignment_and_reversed_time(self):
        row = [34200, 1, 1, 10, 1000000, 1]
        book = [1000100, 20, 1000000, 10]
        self.write([row, row], [book])
        with self.assertRaisesRegex(ValueError, "row counts"):
            self.convert()
        self.assertFalse(self.out.exists())
        self.write([row, [34199, *row[1:]]], [book, book])
        with self.assertRaisesRegex(ValueError, "ordered"):
            self.convert()


if __name__ == "__main__":
    unittest.main()
