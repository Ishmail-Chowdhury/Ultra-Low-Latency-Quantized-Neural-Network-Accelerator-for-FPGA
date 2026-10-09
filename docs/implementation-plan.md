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
6. **Complete — Temporal model evaluation:** compare balanced/unbalanced models
   on three expanding folds, reserve a final test interval, export the selected
   compact model, and verify source replay and saved-memory RTL.
7. **Complete — Real-data development:** reconstruct four complete Nasdaq days,
   investigate scaling, class balance, features, and horizons, freeze selection,
   and evaluate a reserved day. A reliable forecasting advantage remains unproven.
8. **Complete — Software fixed-point frontend:** export integer feature and
   normalization constants, compare predictions with floating preprocessing,
   and drive existing MLP RTL with sampled fixed-feature vectors. See
   [fixed-point preprocessing](fixed-features.md).
9. **Next on macOS:** collect fresh days for rolling evaluation and stronger
   baselines; implement synchronous weight memories and frontend RTL; expand
   simulation coverage. Keep future final-test data isolated from tuning.
10. **Windows/Linux with Vivado, no board required:** synthesize and implement
    for the KR260, constrain clocks, and inspect utilization, timing, and estimated
    power. Actual board latency, throughput, and power measurements require hardware.

Each completed milestone receives a separate Git commit after its checks pass.
The first implementation uses static weights and scales and one clock domain.
Do not interpret simulation cycle counts as achieved hardware clock frequency.

The first LOB specification uses five levels, four history snapshots, a five-event
prediction horizon, and a half-tick stationary band. Preprocessing fits training
data only; feature and label windows remain within chronological partitions.
Synthetic data tests the application pipeline without establishing real-market
performance. See `docs/lob-data.md` and `docs/lob-model.md` for the exact contract.
