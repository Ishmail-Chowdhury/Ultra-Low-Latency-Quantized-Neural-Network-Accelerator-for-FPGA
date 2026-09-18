"""Convert paired LOBSTER message/orderbook CSVs to the project's raw snapshot schema."""

import argparse
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import zip_longest
import json
from pathlib import Path
import tempfile

from models.lob.data import columns, read_snapshots
from scripts.fetch_itch_sample import sha256_file


def convert(message_file, book_file, output, symbol, day, provenance, levels=5):
    date.fromisoformat(day)
    if levels < 1 or not symbol.strip() or not provenance.strip():
        raise ValueError("Positive levels, symbol, and provenance are required")
    if output.resolve() in (message_file.resolve(), book_file.resolve()):
        raise ValueError("Output must differ from the source files")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = skipped = adjusted = segments = source_rows = 0
    previous_time = Decimal(-1)
    last = -1
    gap = True
    trading = True
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "snapshots.csv"
        with message_file.open(newline="") as ms, book_file.open(newline="") as bs, staged.open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(columns(levels))
            for index, (message, book) in enumerate(zip_longest(csv.reader(ms), csv.reader(bs)), 1):
                if message is None or book is None:
                    raise ValueError("LOBSTER message/orderbook row counts differ")
                if len(message) != 6 or len(book) < 4 * levels or len(book) % 4:
                    raise ValueError(f"Invalid LOBSTER columns at row {index}")
                try:
                    seconds = Decimal(message[0])
                    event, reference, size, price, direction = map(int, message[1:])
                    values = list(map(int, book))
                except (ValueError, InvalidOperation) as exc:
                    raise ValueError(f"Invalid LOBSTER numeric field at row {index}") from exc
                if not seconds.is_finite() or not 0 <= seconds < 86400 or seconds < previous_time:
                    raise ValueError("LOBSTER times must be finite, ordered seconds since midnight")
                ns = seconds * 10**9
                if ns != ns.to_integral_value():
                    raise ValueError("LOBSTER timestamp has sub-nanosecond precision")
                if event not in range(1, 8) or reference < 0 or size < 0 or direction not in (-1, 1):
                    raise ValueError("Invalid LOBSTER message fields")
                previous_time = seconds
                source_rows += 1
                if event == 7:
                    if price not in (-1, 0, 1):
                        raise ValueError("Unknown LOBSTER trading-halt marker")
                    trading = price == 1
                asks, ask_sizes, bids, bid_sizes = (values[offset:4 * levels:4] for offset in range(4))
                if any(s < 0 for s in ask_sizes + bid_sizes):
                    raise ValueError("Negative LOBSTER displayed size")
                incomplete = any(p <= 0 or abs(p) == 9999999999 for p in asks + bids) or any(s <= 0 for s in ask_sizes + bid_sizes)
                crossed = bids[0] >= asks[0]
                if event in (6, 7) or not trading or incomplete or crossed:
                    skipped += 1
                    gap = True
                    continue
                if any(a >= b for a, b in zip(asks, asks[1:])) or any(a <= b for a, b in zip(bids, bids[1:])):
                    raise ValueError("LOBSTER price levels are not strictly ordered")
                if gap:
                    segments += 1
                    gap = False
                timestamp = max(int(ns), last + 1)
                adjusted += timestamp != int(ns)
                last = timestamp
                writer.writerow([timestamp, f"{day}/{segments}", symbol] + [
                    v for i in range(levels) for v in
                    (bids[i] / 10000, bid_sizes[i], asks[i] / 10000, ask_sizes[i])])
                count += 1
        read_snapshots(staged, levels)
        metadata = {"schema_version": 1, "dataset_kind": "real", "format": "lobster",
                    "source": provenance, "source_sha256": {"message": sha256_file(message_file), "orderbook": sha256_file(book_file)},
                    "sha256": sha256_file(staged), "levels": levels, "tick_size": 0.01,
                    "symbol": symbol, "date": day, "source_rows": source_rows, "rows": count,
                    "skipped_rows": skipped, "sessions": segments, "timestamp_adjustments": adjusted,
                    "timestamp_basis": "nanoseconds since exchange-local midnight; event-order ties incremented by 1 ns",
                    "gap_policy": "new session after auction, halt, missing depth or locked/crossed book"}
        staged.replace(output)
        output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--orderbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(convert(args.messages, args.orderbook, args.output, args.symbol,
                             args.date, args.source, args.levels), indent=2))


if __name__ == "__main__":
    main()
