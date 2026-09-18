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

`INPUTS`, `OUTPUTS`, `ACC_WIDTH`, `SHIFT`, and `RELU` are elaboration parameters.
`WEIGHT_FILE` and `BIAS_FILE` name `$readmemh` files produced by the exporter.
Weights are row-major. Dimensions may be one or a non-power of two.

The initial memories have asynchronous read ports. This is deliberately a small
functional baseline, **not a claim of inferred BRAM**. A BRAM implementation
needs synchronous reads and corresponding pipeline/control changes. Vivado
synthesis must determine actual resource use before making FPGA claims.

## Cycle accounting

For each of M outputs: one bias-load cycle, N MAC cycles, one requantization
cycle, and one output-transfer cycle. With continuously available inputs and an
always-ready output:

- Final input transfer → final output transfer: `M * (N + 3)` clock periods.
- First input transfer → final output transfer: `N - 1 + M * (N + 3)` periods.
- First input of one vector → first input of the next: `N + M * (N + 3)` periods.

These are edge-to-edge differences. For N=8, M=4 they are 44, 51, and 52 cycles.
Input gaps increase the first-input latency; output stalls increase both latency
measurements. Finite fixed latency is claimed only for the no-stall case. Convert
cycles to time only after a hardware clock frequency is established.
