"""Reconstruct a single Nasdaq ITCH 5.0 instrument into the snapshot CSV contract."""

import argparse
import csv
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from models.lob.data import columns, read_snapshots
from models.lob.itch import ItchBook, messages
from scripts.fetch_itch_sample import sha256_file


def convert(source: Path, output: Path, symbol: str, day: str, provenance: str, levels=5):
    date.fromisoformat(day)
    if levels < 1 or not provenance.strip() or source.resolve() == output.resolve():
        raise ValueError("Positive depth, provenance, and a separate output are required")
    book = ItchBook(symbol)
    count = skipped = adjusted = segments = 0
    last = -1
    gap = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "snapshots.csv"
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rb") as stream, staged.open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(columns(levels))
            for body in messages(stream):
                timestamp, changed = book.apply(body)
                if not changed:
                    continue
                row = book.snapshot(timestamp, day, levels)
                if row is None:
                    skipped += 1
                    gap = True
                    continue
                if gap:
                    segments += 1
                    gap = False
                normalized = max(timestamp, last + 1)
                adjusted += normalized != timestamp
                last = normalized
                writer.writerow([normalized, f"{day}/{segments}", symbol] + [
                    value for i in range(levels) for value in
                    (row.bid_prices[i], row.bid_sizes[i], row.ask_prices[i], row.ask_sizes[i])])
                count += 1
        read_snapshots(staged, levels)  # Fail before publishing malformed output.
        metadata = {"schema_version": 1, "dataset_kind": "real", "format": "nasdaq_itch50",
                    "source": provenance, "source_sha256": sha256_file(source),
                    "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(), "levels": levels,
                    "tick_size": 0.01, "symbol": symbol, "date": day, "rows": count,
                    "timestamp_basis": "nanoseconds since exchange-local midnight; event-order ties incremented by 1 ns",
                    "timestamp_adjustments": adjusted, "sessions": segments,
                    "skipped_incomplete_locked_crossed_or_halted": skipped,
                    "message_counts": dict(book.counts), "end_of_messages_seen": book.ended,
                    "sampling": "one snapshot per target displayed-book update, including updates beyond selected depth",
                    "gap_policy": "new session after invalid depth/book or trading halt"}
        staged.replace(output)
        output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output, args.symbol, args.date, args.source, args.levels), indent=2))


if __name__ == "__main__":
    main()
