"""Development diagnostics and paired, time-blocked accuracy uncertainty."""

import numpy as np


def input_diagnostics(raw, scaler, exponent, names):
    values = np.asarray(raw, dtype=np.float64)
    normalized = (values - np.asarray(scaler.mean)) / np.asarray(scaler.scale)
    bounded = np.clip(normalized, -scaler.clip, scaler.clip) if scaler.clip else normalized
    scaled = bounded / (2.0 ** exponent)
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    saturated = (rounded < -128) | (rounded > 127)
    clipped = np.abs(normalized) > scaler.clip if scaler.clip else np.zeros_like(values, dtype=bool)
    per_feature = [{"feature": name, "int8_saturation_fraction": float(np.mean(saturated[:, i])),
                    "preprocessing_clip_fraction": float(np.mean(clipped[:, i])),
                    "normalized_abs_p99": float(np.quantile(np.abs(normalized[:, i]), .99))}
                   for i, name in enumerate(names)]
    return {"examples": len(values), "input_scale_exponent": exponent,
            "int8_saturation_fraction": float(np.mean(saturated)),
            "examples_with_int8_saturation_fraction": float(np.mean(np.any(saturated, axis=1))),
            "values_outside_int8_range_fraction": float(np.mean((scaled < -128) | (scaled > 127))),
            "preprocessing_clip_fraction": float(np.mean(clipped)),
            "worst_features": sorted(per_feature, key=lambda f: (f["int8_saturation_fraction"],
                                                               f["preprocessing_clip_fraction"],
                                                               f["normalized_abs_p99"]), reverse=True)[:12]}


def paired_accuracy_interval(labels, predictions, baseline, timestamps, seed=2030, repetitions=2000):
    """Resample 60-second blocks, preserving paired predictions within a block.

    This describes uncertainty within the observed test day, not variability
    across unseen days. The last block can have fewer samples.
    """
    labels, predictions, baseline = map(np.asarray, (labels, predictions, baseline))
    if not len(labels) or not (labels.shape == predictions.shape == baseline.shape == (len(timestamps),)):
        raise ValueError("paired metrics require equal nonempty sequences")
    blocks = (np.asarray(timestamps, dtype=np.int64) - timestamps[0]) // (60 * 10**9)
    _, index = np.unique(blocks, return_inverse=True)
    differences = (predictions == labels).astype(float) - (baseline == labels).astype(float)
    sums = np.bincount(index, weights=differences)
    counts = np.bincount(index)
    draws = np.random.default_rng(seed).integers(len(counts), size=(repetitions, len(counts)))
    deltas = np.sum(sums[draws], axis=1) / np.sum(counts[draws], axis=1)
    return {"accuracy_delta_percentage_points": float(np.mean(differences) * 100),
            "interval_95_percentage_points": (np.quantile(deltas, [.025, .975]) * 100).tolist(),
            "block_seconds": 60, "blocks": len(counts), "resamples": repetitions, "seed": seed,
            "scope": "descriptive within-day block bootstrap; not across-day generalization"}
