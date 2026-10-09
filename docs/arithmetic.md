# Integer arithmetic contract (schema version 1)

## Representation and scaling

Inputs, weights, and outputs are signed 8-bit two's-complement integers in
`[-128, 127]`. Zero points are zero. Biases and accumulators use a signed width
`A` (default 32, minimum 16). All values in exported JSON are signed integers;
memory files store their two's-complement bit patterns in hexadecimal.

For an input scale `s_x`, weight scale `s_w`, and output scale `s_y`, the real
values are `x_int * s_x`, `w_int * s_w`, and `y_int * s_y`. Bias integers have
scale `s_x * s_w`. Version 1 requires `s_x * s_w / s_y = 2**(-shift)` with a
static integer `0 <= shift < A`. It uses no general integer multiplier for
requantization. The synthetic fixture assigns no physical units to these scales.

Weights are row-major: `weights[output][input]`.

```text
acc[j] = bias[j] + sum(x[i] * weights[j][i], i=0..N-1)
rounded[j] = round_ties_away_from_zero(acc[j] / 2**shift)
y[j] = clamp(rounded[j], 0 if relu else -128, 127)
```

There is no output softmax. Classification may use argmax of the final logits.
ReLU is optional; it is disabled on a classification output layer.

## Rounding and saturation

For shift zero, the accumulator passes directly to saturation. Otherwise:

```text
magnitude = (abs(acc) + 2**(shift - 1)) >> shift
rounded = -magnitude if acc < 0 else magnitude
```

Examples with shift 1: `3 → 2`, `1 → 1`, `-1 → -1`, `-3 → -2`.
Do not substitute an arithmetic right shift for this operation: negative values
would round differently. RTL must widen before negating the most-negative
accumulator. Saturation happens after rounding; it never wraps to INT8.

## Accumulator bounds

An INT8 product ranges from `-16256` to `16384` and fits signed 16 bits.
The model validator requires, for each output row:

```text
abs(bias[j]) + 128 * sum(abs(weights[j][i])) <= 2**(A - 1) - 1
```

This conservative bound protects **every partial sum** for any legal input,
including bias-first accumulation. It deliberately rejects some safe asymmetric
corner cases. For unknown weights, `abs(bias) + N * 16384` is a sufficient bound.
An N=8 zero-bias worst-case dot product needs at least 19 signed bits; 32 bits
leaves room for larger layers and bias, subject to the same validation.

Accumulator overflow is an error, not a saturation policy. Python checks the
bounds and every partial sum; RTL simulation must fail on a MAC overflow.
Hardware relies on validated parameters and inputs; simulation assertions are
not a runtime hardware error interface.

## Model and fixture format

`model.json` records schema version, matrix dimensions, data/accumulator widths,
shift, ReLU, weights, biases, and rounding/zero-point conventions.
`vectors.json` records the seed, inputs, expected pre-requantization accumulators,
and expected outputs. A fixed seed produces identical files.

`weights.mem` contains N weights for output 0, then output 1, and so on.
`biases.mem` contains one A-bit bias per output. File paths are supplied by the
caller; no machine-specific paths are stored in the model.

The optional software feature frontend has a separate [fixed-point contract](fixed-features.md).
Its Q24 features are normalized and converted to the INT8 inputs described here.
