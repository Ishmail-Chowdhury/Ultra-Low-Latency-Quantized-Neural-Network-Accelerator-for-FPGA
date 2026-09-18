# Quantized Neural-Network FPGA Accelerator

A SystemVerilog accelerator project starting with a bit-exact INT8 dense layer.
The intended FPGA target is the AMD Kria KR260. Limit-order-book prediction is a
potential application; the initial arithmetic benchmark does not establish
prediction accuracy or trading performance.

## Current milestone

- Explicit signed integer arithmetic contract.
- Dependency-free Python dense-layer reference and deterministic vector export.
- Unit tests for rounding, saturation, dimensions, and overflow bounds.

The first verification shape is **8 inputs → 4 outputs**, batch size one, signed
INT8 weights and activations, signed INT32 biases/accumulators, and a static
power-of-two requantization scale. It is a correctness fixture, not a trained model.

## Run

Python 3.10 or newer is required. Run commands from the repository root.

```sh
make test-python
make vectors
```

`make vectors` writes a model, test vectors, expected accumulators/outputs, and
hardware memory files to `build/fixture/`. Generated files are excluded from Git.
The reference and its tests require no third-party Python packages.

See [the arithmetic contract](docs/arithmetic.md) and
[the implementation plan](docs/implementation-plan.md).

## Scope

Start with a reference model, one verified MAC, a sequential dense layer, and
ready/valid flow control. Then add a small trained MLP, synthesize for the KR260,
and measure hardware behavior. Additional precisions, sparsity, DMA, embedded
Linux, and Ethernet are later work. No FPGA timing, resource, or latency claims
are established by software simulation.

License selection is pending; no third-party source code is vendored.
