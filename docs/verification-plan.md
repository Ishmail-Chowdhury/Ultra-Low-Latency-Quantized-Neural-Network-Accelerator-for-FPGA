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
- Early or missing input `tlast` must trigger the framing assertion.

## Trained network

`make benchmark` trains the documented digit MLP, checks the accuracy acceptance
gate, and scores all 450 held-out samples through both RTL layers individually
and through `mlp.sv`. Every case is repeated with source gaps and output stalls.
The network suite also covers inter-layer stalls and reset recovery using a small
synthetic MLP with a deliberately slow output layer.

`make test` runs 38 Python and 11 RTL test methods (many parameterized cases).
The real-data benchmark is a separate command so the core tests remain free of
third-party Python dependencies. GitHub Actions runs the combined audit below; the workflow
configuration has not yet been exercised on GitHub.

## LOB data and application

The Python tests verify CSV validation, seed reproducibility, threshold labels,
causal feature windows, and disjoint raw-row support across chronological splits.
Mutating validation/test data must not change the fitted training standardizer.
Session boundaries must not be crossed by feature or label windows. Additional
checks cover fixed-class metrics, model checksums, replay, and prediction from
prefixes that contain no future rows.

`make lob-benchmark` separately trains the synthetic benchmark and scores all
592 held-out examples through individual RTL layers and the connected MLP, with
and without external stalls. Raw-CSV replay must also reproduce every exported
test prediction. The final-row inference check works without a future label.

## Full software audit

`make validate-software` rebuilds both model benchmarks in an isolated directory.
It runs the regression, repeats LOB training, validates saved memory files,
recomputes chronological labels/splits and training preprocessing, and audits
the written prediction CSV. See [software validation](software-validation.md)
for coverage, reports, and limits. CI runs this combined command; hosted CI has
not yet been exercised here.

## Tool references

- [Icarus command-line flags](https://steveicarus.github.io/iverilog/usage/command_line_flags.html)
- [Icarus simulation](https://steveicarus.github.io/iverilog/usage/simulation.html)

Validated locally with Icarus Verilog 13.0. No vendor-specific primitives or
external verification framework are required by these tests.
