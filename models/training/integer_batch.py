"""NumPy batching of the existing signed INT8/INT32 contract for model selection."""

import numpy as np


def quantize_inputs(values, exponent):
    scaled = np.asarray(values, dtype=np.float64) / (2.0 ** exponent)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Nonfinite quantization input")
    return np.clip(np.sign(scaled) * np.floor(np.abs(scaled) + 0.5), -128, 127).astype(np.int64)


def dense_batch(spec, values):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != spec.inputs or
            not np.issubdtype(values.dtype, np.integer) or np.any(values < -128) or np.any(values > 127)):
        raise ValueError("Expected a matrix of signed INT8 inputs")
    accumulators = values.astype(np.int64) @ np.asarray(spec.weights, dtype=np.int64).T + np.asarray(spec.biases, dtype=np.int64)
    if np.any(accumulators < -(1 << (spec.acc_bits - 1))) or np.any(accumulators >= (1 << (spec.acc_bits - 1))):
        raise ValueError("Accumulator overflow")
    if spec.shift:
        rounded = np.sign(accumulators) * ((np.abs(accumulators) + (1 << (spec.shift - 1))) >> spec.shift)
    else:
        rounded = accumulators
    return accumulators, np.clip(rounded, 0 if spec.relu else -128, 127).astype(np.int64)


def mlp_batch(first, second, values, input_exponent):
    inputs = quantize_inputs(values, input_exponent)
    hidden_acc, hidden = dense_batch(first, inputs)
    output_acc, outputs = dense_batch(second, hidden)
    return inputs, hidden_acc, hidden, output_acc, outputs
