"""Power-of-two, symmetric per-layer quantization for the initial RTL."""

import math
from typing import Sequence

from models.quantization.dense_reference import DenseSpec, checked_int, signed_range


def round_away(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("quantization requires finite values")
    magnitude = math.floor(abs(value) + 0.5)
    return -magnitude if value < 0 else magnitude


def scale_exponent(max_abs: float) -> int:
    """Smallest power-of-two scale whose positive INT8 range covers max_abs."""
    if not math.isfinite(max_abs) or max_abs < 0:
        raise ValueError("calibration maximum must be finite and nonnegative")
    return math.ceil(math.log2(max_abs / 127)) if max_abs else 0


def quantize_value(value: float, exponent: int, bits: int = 8) -> int:
    low, high = signed_range(bits)
    return max(low, min(high, round_away(value / (2.0 ** exponent))))


def quantize_dense(weights: Sequence[Sequence[float]], biases: Sequence[float],
                   input_exponent: int, output_exponent: int, relu: bool) -> tuple[DenseSpec, dict]:
    if not weights or not weights[0]:
        raise ValueError("weight matrix must not be empty")
    weight_exponent = scale_exponent(max(abs(w) for row in weights for w in row))
    bias_exponent = input_exponent + weight_exponent
    shift = output_exponent - bias_exponent
    spec = DenseSpec(
        tuple(tuple(quantize_value(w, weight_exponent) for w in row) for row in weights),
        tuple(checked_int(round_away(b / (2.0 ** bias_exponent)), 32, "quantized bias")
              for b in biases), shift=shift, relu=relu,
    )
    return spec, {
        "input_scale_exponent": input_exponent,
        "weight_scale_exponent": weight_exponent,
        "bias_scale_exponent": bias_exponent,
        "output_scale_exponent": output_exponent,
    }
