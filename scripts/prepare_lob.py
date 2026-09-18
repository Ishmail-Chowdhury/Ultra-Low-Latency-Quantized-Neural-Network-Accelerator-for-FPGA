"""Import snapshot CSV and create disjoint chronological training/validation/test examples."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from models.lob.data import CLASSES, FeatureConfig, feature_names, prepare, read_snapshots


def build(source: Path, output: Path, config: FeatureConfig, kind: str, provenance: str) -> dict:
    if kind not in ("synthetic", "real") or not provenance.strip():
        raise ValueError("provide dataset kind and nonempty provenance")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata_file = source.with_suffix(".metadata.json")
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else None
    if metadata is not None:
        if metadata.get("dataset_kind") != kind or metadata.get("sha256") != source_hash:
            raise ValueError("source metadata kind or checksum mismatch")
        if metadata.get("levels") != config.levels or metadata.get("tick_size") != config.tick_size:
            raise ValueError("source depth or tick size does not match feature configuration")
    rows = read_snapshots(source, config.levels)
    splits = prepare(rows, config)
    document = {"schema_version": 1, "dataset_kind": kind, "provenance": provenance,
                "source_sha256": source_hash, "source_metadata": metadata,
                "symbol": rows[0].symbol, "config": asdict(config),
                "feature_names": feature_names(config), "classes": CLASSES, "splits": splits}
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(json.dumps(document, indent=2) + "\n")
    summary = {"schema_version": 1, "dataset_kind": kind, "source_sha256": source_hash,
               "raw_rows": len(rows), "features": len(document["feature_names"]), "config": asdict(config),
               "splits": {name: {"samples": len(split["examples"]),
                                  "raw_start": split["raw_start"], "raw_stop": split["raw_stop"],
                                  "class_counts": {label: sum(e["label"] == i for e in split["examples"])
                                                   for i, label in enumerate(CLASSES)}}
                          for name, split in splits.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/lob-data/prepared"))
    parser.add_argument("--kind", choices=["synthetic", "real"], required=True)
    parser.add_argument("--source", required=True, help="Dataset provenance, provider, or generator")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--threshold-ticks", type=float, default=0.5)
    args = parser.parse_args()
    config = FeatureConfig(args.levels, args.lookback, args.horizon, args.tick_size, args.threshold_ticks)
    summary = build(args.input, args.output, config, args.kind, args.source)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
