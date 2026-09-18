"""Generate synthetic snapshots for pipeline testing, not market-performance evidence."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random

from models.lob.data import columns


def generate(path: Path, rows: int = 3000, levels: int = 5, seed: int = 2026) -> dict:
    if rows < 1 or levels < 1:
        raise ValueError("rows and levels must be positive")
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    mid_ticks, pressure = 10000, 0.0
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns(levels))
        for index in range(rows):
            pressure = 0.9 * pressure + rng.gauss(0, 0.6)
            imbalance = math.tanh(pressure)
            spread = rng.choice([2, 2, 2, 4])
            values = [index * 1_000_000, "synthetic-session", "SYNTH"]
            for level in range(levels):
                bid = (mid_ticks - spread // 2 - level) * 0.01
                ask = (mid_ticks + spread // 2 + level) * 0.01
                bid_size = max(1, round(100 * (1 + 0.65 * imbalance) + rng.gauss(0, 15)))
                ask_size = max(1, round(100 * (1 - 0.65 * imbalance) + rng.gauss(0, 15)))
                values.extend([f"{bid:.2f}", bid_size, f"{ask:.2f}", ask_size])
            writer.writerow(values)
            # Only after recording t, choose the next price using current pressure.
            draw = rng.random()
            p_down, p_up = 0.2 - 0.18 * imbalance, 0.2 + 0.18 * imbalance
            mid_ticks += -1 if draw < p_down else 1 if draw < p_down + p_up else 0
            mid_ticks = max(levels + 10, mid_ticks)
    metadata = {"schema_version": 1, "dataset_kind": "synthetic", "seed": seed,
                "rows": rows, "levels": levels, "tick_size": 0.01,
                "generator": "pressure-driven toy snapshots; no exchange matching engine",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/lob-data/snapshots.csv"))
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = generate(args.output, args.rows, args.levels, args.seed)
    print(f"Generated {result['rows']} SYNTHETIC snapshots: {args.output}")


if __name__ == "__main__":
    main()
