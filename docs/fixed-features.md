# Fixed-point feature preprocessing

The optional software frontend implements all existing LOB features, including
midpoint changes, queue imbalance, microprice displacement, and sampled flow.
It emits INT8 vectors for the existing dense-layer RTL. The default inference
path remains floating-point preprocessing. Neither model weights nor training
normalization are refitted.

## Arithmetic contract

- Decimal ingress parses prices and quantities exactly in units of 0.0001.
  Sub-unit values are rejected, not silently rounded. Prices are unsigned 32-bit,
  quantities unsigned 48-bit; both must be positive. Integer snapshots can be
  supplied directly by an upstream feed adapter.
- Features and normalized values are signed 64-bit integers with 24 fractional
  bits. A value `v` represents `v / 2**24`. Midpoints retain half-price-unit
  precision through doubled integer prices.
- Ratios use integer division rounded to nearest, ties away from zero. Signed
  128-bit intermediate bounds are checked. Overflow is an error, not wrapping.
- `log1p(quantity)` uses integer exponent extraction, a Q32 mantissa, and linear
  interpolation in a 1,025-entry Q30 table for `ln(1 + i/1024)`. Decimal logarithms
  are evaluated only during offline export. Runtime logarithms use table reads,
  multiplication, addition, and shifts.
- Normalization subtracts an exported Q24 center, multiplies by an unsigned
  64-bit Q30 reciprocal scale, then rounds back to Q24. Optional trained clipping
  bounds are applied afterward. Extremely small/large training scales that cannot
  be represented by these constants are explicitly rejected.
- Conversion to the model's INT8 scale rounds by `24 + input_scale_exponent`
  bits and saturates to [-128,127]. Supported input exponents are -24..31.
- Supported shape: 1..32 levels and 1..64 history snapshots. Session boundaries,
  instrument identity, strictly increasing timestamps, and valid books are checked.

`FixedSnapshot` marks the integer runtime boundary. CSV parsing uses Decimal;
constant compilation uses Decimal and existing floating training metadata.
Feature arithmetic after this boundary uses integers only. Integer division and
wide intermediates are a reference contract, not a claim of a cheap, pipelined
hardware implementation. The feature frontend itself has **not** been written
in RTL, synthesized, or timed.

## Export, validate, and replay

```sh
python3 -m scripts.export_fixed_frontend --model build/prediction-quality \
  --output build/fixed-features/validated-frontend
python3 -m scripts.validate_fixed_frontend --rtl-samples 256
python3 -m scripts.predict_lob --model build/prediction-quality \
  --input build/nasdaq-days/snapshots.csv \
  --frontend build/fixed-features/validated-frontend \
  --output build/fixed-features/predictions.csv
```

Exports require an empty output directory outside the frozen model. `frontend.json`
binds constants to the model report hash. `centers.mem`, `inverse_scales.mem`, and
`log_lut.mem` provide ROM words. Loading recomputes constants from the checksum-
validated model and rejects any metadata or ROM mismatch.

The validation command compares every saved held-out input and prediction against
floating preprocessing. It records feature/normalization error, changed INT8
values, changed logits and predictions, and both accuracy reports. Optional RTL
checks feed uniformly sampled fixed-feature vectors into the existing MLP with
and without stalls. These checks validate the MLP boundary, not frontend RTL.
The old test day is used only for arithmetic fidelity; this is not fresh model
selection or evidence of a new forecasting advantage.

Unit tests cover signed ties, overflow and saturation, decimal precision, lookup
accuracy against high-precision Decimal logarithms, every feature mode, half-tick
midpoints, session isolation, causal prefix equivalence, integer-only runtime,
export binding, ROM corruption, and inference integration.

## Recorded validation

All 82 Python and 15 RTL regression tests pass. On all 23,377 saved test
examples, 0 predictions and 0 logit vectors changed.
0 of 2384454 INT8 input values changed; maximum difference is 0.
Maximum raw feature error is 1.36726e-07 and maximum normalized
error is 3.06806e-07. Accuracy is 45.5277% with fixed preprocessing,
compared with 45.5277% with floating preprocessing. The 256 sampled fixed-feature
vectors pass existing MLP RTL with and without stalls.
See [the saved validation report](../results/fixed-features/validation.json).
