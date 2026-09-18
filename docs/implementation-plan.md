# Implementation milestones

1. **Arithmetic and reference:** specify signed arithmetic, write a Python dense
   layer, export deterministic fixtures, and check boundaries.
2. **Compute blocks:** verify a signed MAC and rounding/saturation block against
   Python, including overflow detection.
3. **Streaming dense layer:** buffer an input vector, reuse one MAC, stream
   outputs, and verify backpressure and reset. Parameterize matrix dimensions.
4. **Trained baseline:** use a documented small benchmark, train a floating-point
   MLP, quantize it to the same arithmetic, and compare accuracy and RTL results.
5. **Hardware:** build for the KR260, establish a clock constraint, inspect timing
   and resource reports, and run identical vectors through a hardware test harness.

Each completed milestone receives a separate Git commit after its checks pass.
The first implementation uses static weights and scales and one clock domain.
Do not interpret simulation cycle counts as achieved hardware clock frequency.

For a later limit-order-book application, first define the data source, features,
lookback, prediction horizon, labels, and chronological splits. Fit preprocessing
only on training data and account for label-window overlap at split boundaries.
The generic dense-layer fixture does not make these application choices.
