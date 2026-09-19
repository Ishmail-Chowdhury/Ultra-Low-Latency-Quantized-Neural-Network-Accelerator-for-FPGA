"""Reconstruct complete ITCH days and sample causal regular-hours books on a grid."""

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import tempfile

from models.lob.data import columns, read_snapshots
from models.lob.itch import ItchBook, messages, uint
from scripts.fetch_itch_sample import sha256_file

OPEN_NS = (9 * 3600 + 30 * 60) * 10**9
CLOSE_NS = 16 * 3600 * 10**9


def sample_day(source, writer, day, symbol, interval_ns=10**9, levels=5,
               open_ns=OPEN_NS, close_ns=CLOSE_NS):
    if interval_ns <= 0 or not 0 <= open_ns < close_ns <= 86400 * 10**9:
        raise ValueError("Invalid sampling interval or session hours")
    book = ItchBook(symbol)
    next_grid = open_ns
    segment = 0
    gap = True
    emitted = skipped = 0
    prior = -1
    system_events = []
    # Day-based monotone coordinate, explicitly not an epoch timezone conversion.
    day_offset = (date.fromisoformat(day) - date(1970, 1, 1)).days * 86400 * 10**9

    def emit_until(timestamp):
        nonlocal next_grid, segment, gap, emitted, skipped
        while next_grid < min(timestamp, close_ns):
            row = book.snapshot(next_grid, day, levels)
            if row is None:
                gap = True
                skipped += 1
            else:
                if gap:
                    segment += 1
                    gap = False
                writer.writerow([day_offset + next_grid, f"{day}/{segment}", symbol] + [
                    v for i in range(levels) for v in
                    (row.bid_prices[i], row.bid_sizes[i], row.ask_prices[i], row.ask_sizes[i])])
                emitted += 1
            next_grid += interval_ns

    with source.open("rb") as stream:
        for body in messages(stream):
            timestamp = uint(body, 5, 6)
            if timestamp < prior:
                raise ValueError("Filtered ITCH stream timestamps decrease")
            prior = timestamp
            # Emit using only updates at or before the grid timestamp. All
            # same-time messages are applied before that timestamp is sampled.
            emit_until(timestamp)
            _, changed = book.apply(body)
            if body[:1] == b"S":
                system_events.append(chr(body[11]))
            if changed and body[:1] == b"H" and not book.trading:
                gap = True  # Even sub-second halts must break feature windows.
    if not book.ended or system_events != ["O", "S", "Q", "M", "E", "C"]:
        raise ValueError("Full-day import requires all six daily system events")
    if next_grid < close_ns:
        raise ValueError("Source ends before regular-hours sampling is complete")
    expected = (close_ns - open_ns + interval_ns - 1) // interval_ns
    return {"date": day, "rows": emitted, "missing_grid_points": skipped,
            "expected_grid_points": expected, "coverage": emitted / expected,
            "sessions": segment, "system_events": system_events,
            "source_sha256": sha256_file(source), "message_counts": dict(book.counts)}


def prepare(sources, output, symbol="AAPL", interval_ns=10**9):
    days = [day for day, _ in sources]
    if len(days) < 4 or days != sorted(set(days)):
        raise ValueError("At least four distinct chronological days are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    audits = []
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "snapshots.csv"
        with staged.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns(5))
            for day, source in sources:
                download = json.loads(source.with_suffix(".download.json").read_text())
                if (not download.get("complete_archive") or not download.get("gzip_trailer_verified") or
                        download.get("date") != day or download.get("symbol") != symbol or
                        download.get("sha256") != sha256_file(source)):
                    raise ValueError("Full-day download provenance does not match source")
                audit = sample_day(source, writer, day, symbol, interval_ns)
                if audit["coverage"] < 0.99:
                    raise ValueError(f"Less than 99% regular-hours coverage on {day}")
                audits.append({**audit, "download": download})
                print(f"{day}: {audit['rows']} snapshots; coverage {audit['coverage']:.3%}", flush=True)
        read_snapshots(staged, 5)
        metadata = {"schema_version": 1, "dataset_kind": "real", "format": "nasdaq_itch50_regular_grid",
                    "source": "Complete public Nasdaq ITCH archives; AAPL; regular-hours causal grid",
                    "symbol": symbol, "levels": 5, "tick_size": 0.01,
                    "sha256": sha256_file(staged), "interval_ns": interval_ns,
                    "regular_hours": "09:30:00 inclusive to 16:00:00 exclusive, exchange local",
                    "timestamp_basis": "calendar-day ordinal since 1970 plus exchange-local ns; NOT UTC epoch",
                    "sampling": "book after all source updates with time <= grid; no future interpolation",
                    "gap_policy": "new session after invalid grid point or trading halt",
                    "days": audits, "split_days": {"train": days[:-2], "validation": days[-2:-1], "test": days[-1:]}}
        staged.replace(output)
        output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/nasdaq-days/raw"))
    parser.add_argument("--output", type=Path, default=Path("build/nasdaq-days/snapshots.csv"))
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--interval-seconds", type=int, default=1)
    args = parser.parse_args()
    sources = [(path.name[:10], path) for path in sorted(args.input.glob(f"*-{args.symbol}.itch"))]
    prepare(sources, args.output, args.symbol, args.interval_seconds * 10**9)


if __name__ == "__main__":
    main()
