# Implementation milestones

1. **Complete — Arithmetic and reference:** specify signed arithmetic, write a Python dense
   layer, export deterministic fixtures, and check boundaries.
2. **Complete — Compute blocks:** verify a signed MAC and rounding/saturation block against
   Python, including overflow detection.
3. **Complete — Streaming dense layer:** buffer an input vector, reuse one MAC, stream
   outputs, and verify backpressure and reset. Parameterize matrix dimensions.
4. **Complete — Trained baseline:** use a documented small benchmark, train a floating-point
   MLP, quantize it to the same arithmetic, and compare accuracy and RTL results.
5. **Complete — Software LOB pipeline:** create synthetic snapshots and import CSVs,
   build disjoint chronological samples, select a compact MLP on validation data,
   and verify integer inference through raw-CSV replay and RTL simulation.
6. **Next — Software experiments:** import real snapshots, add walk-forward
   validation, investigate class balance, and specify fixed-point preprocessing.
7. **When hardware is available:** build for the KR260, establish a clock constraint, inspect timing
   and resource reports, and run identical vectors through a hardware test harness.

Each completed milestone receives a separate Git commit after its checks pass.
The first implementation uses static weights and scales and one clock domain.
Do not interpret simulation cycle counts as achieved hardware clock frequency.

The first LOB specification uses five levels, four history snapshots, a five-event
prediction horizon, and a half-tick stationary band. Preprocessing fits training
data only; feature and label windows remain within chronological partitions.
Synthetic data tests the application pipeline without establishing real-market
performance. See `docs/lob-data.md` and `docs/lob-model.md` for the exact contract.
