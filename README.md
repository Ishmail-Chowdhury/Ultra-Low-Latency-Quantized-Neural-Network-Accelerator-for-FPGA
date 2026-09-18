# Quantized Neural-Network FPGA Accelerator

A SystemVerilog accelerator project starting with a bit-exact INT8 dense layer.
The intended FPGA target is the AMD Kria KR260. Development currently runs without
hardware: Python training, integer inference, and RTL simulation. A synthetic
limit-order-book pipeline now exercises the intended application; it does not
establish real-market prediction accuracy or trading performance.

## Current milestone

- Explicit signed integer arithmetic contract.
- Dependency-free Python dense-layer reference and deterministic vector export.
- Unit tests for rounding, saturation, dimensions, and overflow bounds.
- Signed MAC and requantization RTL, verified against the Python arithmetic.
- Parameterized streaming dense layer with backpressure and reset regression tests.
- Configurable parallel neuron lanes with bit-exact results and measured simulation cycles.
- Two-layer streaming MLP and a reproducible trained digit-classification benchmark.
- Synthetic order-book data, a strict CSV importer, chronological splits, and an
  88→16→3 direction classifier with software replay.

**Initial software workflow:** `make lob-benchmark` generates 3,000 synthetic
snapshots, trains and quantizes the model, verifies its RTL, and writes prediction
CSV files. No board is required. See [the data contract](docs/lob-data.md),
[model commands](docs/lob-model.md), and [synthetic results](docs/lob-results.md).

**Expanding-window evaluation:** `make walk-forward` compares balanced and
unbalanced MLPs and linear models on three successive validation periods, then
tests the selected models on a reserved final interval. The recorded selected
MLP is 88→8→3. See [the protocol and results](docs/walk-forward.md).

**Improvement experiment:** `make improve-lob` selects using INT8 validation
scores, stronger regularization, and a stationary-output adjustment. On the same
fresh synthetic test examples, stationary recall rises from 15.70% to 22.31%
and macro-F1 from 0.4545 to 0.4744. The selected 88→16→3 model costs more cycles.
See [the comparison and trade-offs](docs/improved-lob.md).

**Parallel RTL:** after generating the saved model with `make improve-lob`, run
`make improve-rtl` to compare one-lane and parallel implementations on the same
592 test vectors. No retraining or board is needed for the RTL comparison.
See [lane configurations, cycle measurements, and resource trade-offs](docs/parallel-rtl.md).

**First real-data test:** Nasdaq ITCH reconstruction produced 990 five-level AAPL
snapshots from a bounded premarket sample. The frozen synthetic model scored
37.88% accuracy versus 42.36% for the stationary baseline, while RTL matched all
982 predictions exactly. A separate chronological real-data training run also
failed to beat its accuracy baseline. See [results, reproduction commands, and
access status for LOBSTER, FI-2010, and Coinbase](docs/real-data-tests.md).

The 64→16→10 benchmark achieves **96.22% floating-point accuracy** and **95.33%
INT8 accuracy** on 450 held-out images. Both RTL layers and the complete network
match the integer reference exactly for every test image, with and without
backpressure. See [recorded results](docs/results.md) and
[benchmark methodology](docs/benchmark.md).

The first verification shape is **8 inputs → 4 outputs**, batch size one, signed
INT8 weights and activations, signed INT32 biases/accumulators, and a static
power-of-two requantization scale. It is a correctness fixture, not a trained model.

## Run

Python 3.10 or newer is required. Run commands from the repository root.

```sh
make test-python
make vectors
make test
```

For a fresh end-to-end software audit, run **`make validate-software`** after
installing the training dependencies. It checks both benchmarks, repeated LOB
training, saved hardware memories, chronological labels/splits, and written CSV
predictions. Results and logs go to `build/software-validation/`.
See [software validation](docs/software-validation.md).

`make vectors` writes a model, test vectors, expected accumulators/outputs, and
hardware memory files to `build/fixture/`. Generated files are excluded from Git.
The reference and its tests require no third-party Python packages.

RTL tests require Icarus Verilog (`iverilog` and `vvp`). Install with
`brew install icarus-verilog` on a supported Homebrew setup, or use a local Conda
environment:

```sh
conda create --prefix ./build/simulator --override-channels -c conda-forge iverilog -y
```

The test runner searches `PATH`, then `build/simulator/bin/`. Override tool
locations with `IVERILOG` and `VVP` if needed. Missing tools fail the RTL checks;
they are not silently skipped. Tests use Python `unittest` to drive self-checking
SystemVerilog benches, so cocotb is not required for this milestone.

To train and verify the real benchmark, install the pinned optional dependencies
in a Python 3.11/3.12 environment and run:

```sh
python3 -m pip install -r requirements-training.txt
make benchmark
```

`make train` and `make verify-model` run those stages separately. GitHub Actions
is configured to run the regression and benchmarks on pushes and pull requests;
the hosted workflow has not yet been run for this repository.

See [the arithmetic contract](docs/arithmetic.md) and
[the implementation plan](docs/implementation-plan.md).
Verification coverage is recorded in [the verification plan](docs/verification-plan.md).
See [streaming interfaces and cycle accounting](docs/interfaces.md) for the
configurable MAC lanes. The default one-lane 8×4 fixture completes 44 cycles after the last input
transfer when the receiver is always ready; this is a simulation result.

## Scope

The reference, compute blocks, streaming layers, and trained models run in
software and simulation. Next software work can use real CSV snapshots, improve
chronological validation, and investigate fixed-point feature preprocessing.
KR260 synthesis and hardware measurements can wait until hardware is available.
Additional precisions, sparsity, DMA, embedded
Linux, and Ethernet are later work. No FPGA timing, resource, or latency claims
are established by software simulation.

License selection is pending; no third-party source code is vendored.
