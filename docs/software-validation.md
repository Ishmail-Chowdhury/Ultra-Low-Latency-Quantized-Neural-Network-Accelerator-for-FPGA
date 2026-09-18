# Software validation without hardware

Run the full validation with the existing Python training dependencies and Icarus:

```sh
make validate-software
```

Each run creates a fresh directory beneath `build/software-validation/`. Existing
models in `build/digits/` and `build/lob/` are not overwritten. The latest
`build/software-validation/report.json` records each stage and its log. The report
starts with `passed: false`; it becomes true only when every check succeeds.
Exceptions and timeouts preserve a failed report and a nonzero exit status.

## Checks

1. Run the Python and RTL regression test methods (currently 44 and 11), including deliberately
   corrupted exports and simulation using modified saved memory files.
2. Generate and prepare a fresh 3,000-row synthetic dataset with chronological
   feature/label separation and strict CSV validation.
3. Train the LOB pipeline twice in separate directories. Compare complete reports
   and all recorded artifact hashes for exact local reproducibility.
4. Compare saved weight/bias memory words against JSON, then simulate the actual
   saved files through each layer and the connected MLP. Score all 592 held-out
   examples both with and without stalls.
5. Rebuild chronological partitions and labels from source snapshots. Check each
   exported index, timestamp, label, split interval, and training-only normalizer.
6. Check quantized inputs, intermediate accumulators, activations, final logits,
   and reported INT8 metrics against raw-source replay.
7. Write 2,997 causal prediction rows. Check their ordering and coverage, and
   verify held-out timestamps, class IDs, and logits in the written CSV. Inference
   also runs on the final input row, which has no future label.
8. Retrain the digit benchmark and verify all 450 held-out examples through its
   actual saved RTL memories as a regression of the original application.

## Gaps fixed during validation

The simulator previously rebuilt memory files from JSON. Export verification now
uses the saved files, so a memory-file error cannot be hidden by regeneration.
Synthetic unit tests may still generate their own fixtures.

A complete checksum manifest is required. Checks establish agreement among
network dimensions, layer arithmetic, adjacent scales, feature ordering,
normalization dimensions, and memory contents. Corruption tests update hashes
when appropriate to demonstrate that semantic checks still catch inconsistencies.

Replay previously compared only final logits. It now checks labels, split
membership, coverage, preprocessing, inputs, intermediate values, and reported
metrics. Matching saturated final outputs alone no longer suffices.

Hashes detect accidental changes; they do not authenticate a model's publisher.
These checks establish software consistency and functional RTL agreement. They
do not establish real-market accuracy, independent statistical generalization,
timing closure, resource usage, or measured FPGA latency. The LOB data remains
synthetic. Reproducibility is checked within the current environment; different
floating-point libraries/platforms may produce another valid exported model.

## Recorded run

The 2026-09-18 local run passed all 49 tests, both saved-memory benchmarks,
source-data replay, and repeated-training comparison. Reference reports are
stored in [results/software-validation](../results/software-validation/report.json).
Detailed run logs remain under the ignored `build/software-validation/` directory.
