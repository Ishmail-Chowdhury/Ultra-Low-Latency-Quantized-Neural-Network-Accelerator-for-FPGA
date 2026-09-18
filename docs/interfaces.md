# Streaming dense layer

`dense_layer` uses a single clock and active-high synchronous reset. Reset must
span a rising edge. It discards the in-flight vector and any pending outputs;
after reset, the sender must restart at element zero. Ready and valid are gated
low while reset is asserted. Weight/bias ROM contents persist through reset.

## Transfers

A transfer occurs on a rising edge with `tvalid && tready` high. Data is signed
INT8. Input elements arrive in ascending feature order, and output elements leave
in ascending neuron order. Input `tlast` must be high exactly on element N-1;
output `tlast` is high on element M-1. Valid/data/last must remain stable until
accepted. This is an internal ready/valid interface, not a complete AXI subsystem.

There is room for one input vector. The module deasserts input ready while
computing and sending that vector. It holds an output indefinitely if the
receiver is not ready. The source can present the next vector during computation
provided it obeys the hold rule. No transfers are dropped under backpressure.

Malformed input framing terminates simulation through an assertion. There is
currently no synthesized error/status interface or recovery from malformed
packets other than reset; hardware integration must supply correctly framed data.

## Static configuration

`INPUTS`, `OUTPUTS`, `ACC_WIDTH`, `SHIFT`, `RELU`, and `LANES` are elaboration parameters.
`WEIGHT_FILE` and `BIAS_FILE` name `$readmemh` files produced by the exporter.
Weights are row-major. Dimensions may be one or a non-power of two.

`LANES` defaults to one and must be between one and `OUTPUTS`. Each lane has
its own signed MAC and requantizer. Lanes share the input element and compute
consecutive neurons in parallel. The final group masks unused lanes when the
output count is not divisible by `LANES`. A result register per lane holds the
group while the output stream serializes it in neuron order. All group results
and stream position remain stable under backpressure; reset discards them.
The saved memory layout and arithmetic contract do not change.

The initial memories have asynchronous read ports. This is deliberately a small
functional baseline, **not a claim of inferred BRAM**. A BRAM implementation
needs synchronous reads and corresponding pipeline/control changes. Vivado
synthesis must determine actual resource use before making FPGA claims.
Parallel lanes require additional independent weight reads and arithmetic units;
the logical lane count is not a measured DSP or LUT count.

## Cycle accounting

For N inputs, M outputs, and L lanes, there are `G = ceil(M / L)` groups.
Each group takes one bias-load cycle, N MAC cycles, and one requantization cycle.
Output serialization takes M transfer cycles in total, including a partial final
group. With continuously available inputs and an always-ready output:

- Final input transfer → final output transfer: `G * (N + 2) + M` clock periods.
- First input transfer → final output transfer: `N - 1 + G * (N + 2) + M` periods.
- First input of one vector → first input of the next: `N + G * (N + 2) + M` periods.

These are edge-to-edge differences. For N=8, M=4, L=1 they are 44, 51, and 52
cycles, preserving the original `M * (N + 3)` compute latency. L=4 reduces these
to 14, 21, and 22 cycles.
Input gaps increase the first-input latency; output stalls increase both latency
measurements. Finite fixed latency is claimed only for the no-stall case. Convert
cycles to time only after a hardware clock frequency is established.

## Two-layer MLP

`rtl/top/mlp.sv` connects two dense layers directly: hidden ReLU followed by signed
output logits. Each layer has its own MAC lanes and input buffer. Set
`HIDDEN_LANES` and `OUTPUT_LANES` independently; both default to one. The hidden layer can
begin accepting another vector while the output layer finishes the preceding one;
its outputs stall if the next layer cannot yet accept them. Reset flushes both.

For a single isolated 64→16→10 vector with one lane per layer, final input to final output takes
`16 * (64 + 3) + 10 * (16 + 3) = 1262` clock periods without external stalls.
Repeated vectors can experience internal stalls even with an always-ready final
receiver. Measure this traffic pattern separately; a single-layer cycle formula
does not characterize the full pipeline's steady-state latency.
For an isolated vector, sum the two layer compute latencies. See
[parallel RTL measurements](parallel-rtl.md) for the LOB model and repeated-vector
input intervals. The scoreboard checks the isolated-vector formula and measures
minimum/maximum latency and input-start intervals across the complete stream.

The RTL consumes static per-layer shift parameters from the export metadata.
There is no runtime model-loading interface or on-chip argmax yet. A host can
select the highest output logit to classify the vector.
