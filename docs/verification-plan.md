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
- Parallel lanes from one through the output count, including partial groups,
  signed extrema, nonzero biases, ReLU, saturation, and extreme shifts.
- Reset after accepting the first output of a group, while the next is blocked.
- Parallel MLP layers with independent lane counts, including a fast hidden layer
  feeding a slower output layer, and isolated-vector cycle-count assertions.

## Trained network

`make benchmark` trains the documented digit MLP, checks the accuracy acceptance
gate, and scores all 450 held-out samples through both RTL layers individually
and through `mlp.sv`. Every case is repeated with source gaps and output stalls.
The network suite also covers inter-layer stalls and reset recovery using a small
synthetic MLP with a deliberately slow output layer.

`make test` runs 59 Python and 15 RTL test methods (many parameterized cases).
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

## Expanding-window experiment

The expanding-window tests verify disjoint feature/label support, growth of
training prefixes, isolation of the final holdout, training-only balancing and
normalization, stable selection ties, and session boundaries. `make walk-forward`
also verifies all selected-model held-out vectors through source replay and RTL.

## Improvement experiment

`make improve-lob` additionally compares the selected INT8 model with the previous
design refitted on identical data. Source replay independently checks baseline
predictions and comparison deltas. Corruption tests reject omitted baseline
checksums, modified baseline scores, and inflated reported improvements.

## Parallel RTL benchmark

`make improve-rtl` uses the saved improved model to compare five lane
configurations. Every configuration verifies every held-out example in both
individual layers and the full network, with and without stalls, using the saved
ROM files. Reports record lane counts, logical MAC instances, latency, input
start intervals, and artifact hashes. See [parallel RTL](parallel-rtl.md).

## Tool references

The [real-data test](real-data-tests.md) separately exercises 982 frozen-model
examples reconstructed from Nasdaq AAPL ITCH messages, plus all 190 held-out
examples of a model trained on that sample. This network-dependent data download
is an explicit local command, not part of the offline unit regression.

- [Icarus command-line flags](https://steveicarus.github.io/iverilog/usage/command_line_flags.html)
- [Icarus simulation](https://steveicarus.github.io/iverilog/usage/simulation.html)

Validated locally with Icarus Verilog 13.0. No vendor-specific primitives or
external verification framework are required by these tests.
