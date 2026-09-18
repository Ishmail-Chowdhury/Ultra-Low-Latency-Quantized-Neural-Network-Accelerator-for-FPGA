# Verification

Run `make test` for Python and RTL checks. RTL tests create temporary benches and
memory files, compile with Icarus `-g2012 -Wall`, and require both a successful
simulator exit and a PASS marker. Simulator processes have timeouts. The tests
use fixed seeds. Run `make test-python` or `make test-rtl` independently if needed.

## Arithmetic reference

- Hand-calculated dense-layer results.
- Ties and neighboring values on both sides of zero.
- INT8 clipping, ReLU, and the most-negative INT32 accumulator.
- Invalid tensor dimensions, types, ranges, shifts, and unsafe accumulator bounds.
- JSON roundtrip, hexadecimal layout, and seed reproducibility.

## Compute RTL

- All 65,536 combinations of signed INT8 operands.
- 2,048 seeded accumulation/hold steps and randomized bias loads.
- Reset/load/enable priority, reset during activity, and accumulator extrema.
- Positive and negative overflow must terminate simulation with the MAC assertion.
- More than 16,000 requantization cases across 16/32-bit accumulators, shift zero,
  shift one, shift seven, the maximum shift, and ReLU on/off.

The bench calculates expected values independently in Python and passes them to
the simulator. Exhaustive multiplication also checks SystemVerilog signedness.
This is functional simulation; synthesis, timing closure, and board validation
remain separate milestones.

## Streaming layer

- All 36 exported fixture vectors, including 144 scored output elements.
- Exact no-stall cycle counts for the 8×4 layer and alternate dimensions.
- Three fixed seeds for source gaps and receiver stalls, including prolonged
  output blocking, with independent checks for stable output data/last.
- Repeated vectors, output order, packet boundaries, and no extra outputs.
- Reset with a partial input, during MAC activity, and while an output is stalled;
  subsequent complete vectors must match the reference.
- 1×1 and non-power-of-two matrices, ReLU, and extreme requantization shifts.

## Tool references

- [Icarus command-line flags](https://steveicarus.github.io/iverilog/usage/command_line_flags.html)
- [Icarus simulation](https://steveicarus.github.io/iverilog/usage/simulation.html)

Validated locally with Icarus Verilog 13.0. No vendor-specific primitives or
external verification framework are required by these tests.
