"""Expanding-window selection with a reserved final 20% interval."""

import math

from models.lob.data import examples_in_range


def plan(row_count: int) -> dict:
    if type(row_count) is not int or row_count < 15:
        raise ValueError("at least 15 raw rows are required; feature windows may require more")
    initial, stop = 2 * row_count // 5, 4 * row_count // 5
    boundaries = [initial + i * (stop - initial) // 3 for i in range(4)]
    return {"schema_version": 1, "raw_rows": row_count, "holdout_start": stop,
            "folds": [{"train_start": 0, "train_stop": a, "validation_start": a, "validation_stop": b}
                      for a, b in zip(boundaries, boundaries[1:])]}


def selection_folds(rows, config) -> list[dict]:
    return [{"train": examples_in_range(rows, config, fold["train_start"], fold["train_stop"]),
             "validation": examples_in_range(rows, config, fold["validation_start"], fold["validation_stop"])}
            for fold in plan(len(rows))["folds"]]


def final_splits(rows, config) -> dict:
    stop = plan(len(rows))["holdout_start"]
    return {"train": examples_in_range(rows, config, 0, stop),
            "test": examples_in_range(rows, config, stop, len(rows))}


def balanced_weights(labels) -> list[float]:
    """Each class contributes equal total weight, using these training labels only."""
    if not labels or set(labels) != {0, 1, 2}:
        raise ValueError("class balancing requires all three training classes")
    counts = [labels.count(i) for i in range(3)]
    return [len(labels) / (3 * counts[label]) for label in labels]


def select_candidate(records: list[dict], family: str) -> dict:
    eligible = [record for record in records if record["family"] == family]
    if not eligible:
        raise ValueError(f"no candidate for family {family}")
    for record in eligible:
        if len(record["fold_metrics"]) != 3 or any(
                not math.isfinite(m["macro_f1"]) for m in record["fold_metrics"]):
            raise ValueError("candidate must have three finite validation scores")
    # Candidate order is declared before any scores are observed and breaks exact ties.
    return max(eligible, key=lambda record: sum(m["macro_f1"] for m in record["fold_metrics"]) / 3)
