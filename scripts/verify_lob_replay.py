"""Check raw-CSV inference against every exported held-out integer test vector."""

import argparse
import hashlib
import json
from pathlib import Path

from models.lob.data import read_snapshots
from models.lob.inference import LobPredictor


def verify(model: Path, source: Path) -> dict:
    predictor = LobPredictor(model)
    report = json.loads((model / "report.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != report["source_sha256"]:
        raise ValueError("replay verification requires the original dataset CSV")
    rows = read_snapshots(source, predictor.config.levels)
    vectors = json.loads((model / "vectors.json").read_text())["vectors"]
    if not vectors:
        raise ValueError("no held-out vectors to verify")
    for vector in vectors:
        index = vector["dataset_index"]
        prediction = predictor.predict(rows, index)
        if prediction["logits"] != vector["outputs"]:
            raise AssertionError(f"Raw feature/preprocessing/export mismatch at snapshot {index}")
    # Inference also works at the end of the file, where no future label is available.
    tail = predictor.predict(rows, len(rows) - 1)
    result = {"dataset_kind": report["dataset_kind"], "samples": len(vectors),
              "raw_csv_matches_export": True, "tail_prediction_without_future_label": tail}
    (model / "replay_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/lob"))
    parser.add_argument("--input", type=Path, default=Path("build/lob-data/snapshots.csv"))
    args = parser.parse_args()
    result = verify(args.model, args.input)
    print(f"PASS: raw CSV inference matches all {result['samples']} held-out exported predictions")


if __name__ == "__main__":
    main()
