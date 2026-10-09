"""Replay snapshot CSV through an exported integer model without future labels or hardware."""

import argparse
import csv
from pathlib import Path

from models.lob.data import read_snapshots
from models.lob.inference import LobPredictor


def replay(model: Path, source: Path, output: Path, *, frontend: Path | None = None) -> int:
    predictor = LobPredictor(model, frontend=frontend)
    if frontend is None:
        rows = read_snapshots(source, predictor.config.levels)
    else:
        from models.lob.fixed_features import read_fixed_snapshots
        rows = read_fixed_snapshots(source, predictor.config.levels)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["decision_index", "timestamp_ns", "class_id", "prediction",
                         "logit_down", "logit_stationary", "logit_up", "training_dataset_kind"])
        for index in range(predictor.config.lookback - 1, len(rows)):
            window = rows[index - predictor.config.lookback + 1:index + 1]
            if any(row.session_id != window[0].session_id for row in window):
                continue
            result = predictor.predict(rows, index)
            writer.writerow([result["decision_index"], result["timestamp_ns"], result["class_id"],
                             result["prediction"], *result["logits"], predictor.network["dataset_kind"]])
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/lob"))
    parser.add_argument("--input", type=Path, default=Path("build/lob-data/snapshots.csv"))
    parser.add_argument("--output", type=Path, default=Path("build/lob/predictions.csv"))
    parser.add_argument("--frontend", type=Path, help="Use a separately exported fixed-point feature frontend")
    args = parser.parse_args()
    count = replay(args.model, args.input, args.output, frontend=args.frontend)
    mode = "fixed-point" if args.frontend else "floating-point"
    print(f"Wrote {count} causal predictions using {mode} preprocessing to {args.output}; logits are integer scores, not probabilities")


if __name__ == "__main__":
    main()
