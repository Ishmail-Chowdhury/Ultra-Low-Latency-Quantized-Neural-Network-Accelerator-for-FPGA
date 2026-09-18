# Synthetic LOB baseline results

This run uses **generated toy snapshots, not real exchange data**. Its purpose is
to verify the data → training → quantization → software/RTL inference pipeline.
It does not establish market predictability, profitability, or trading latency.

Run date: 2026-09-18. Seed: 2026. Source: 3,000 five-level synthetic snapshots.
Chronological examples: 1,792 training, 592 validation, 592 test. Each example
uses four snapshots of history and predicts endpoint mid-price movement five
snapshots ahead, with a half-tick stationary band.

Validation macro-F1 selected **88→16→3** over 88→8→3 (0.4849 versus 0.4628).
Model selection and calibration used no test data. The model uses hidden ReLU,
INT8 weights/activations, INT32 accumulators, and one MAC per RTL layer.

| Held-out test classifier | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Training-majority class | 44.09% | 0.2040 |
| Always stationary | 20.44% | 0.1131 |
| Logistic regression | 58.78% | 0.4426 |
| Floating-point MLP | 56.08% | 0.4515 |
| INT8 MLP | 55.91% | 0.4469 |

INT8 reduces accuracy by 0.1689 percentage points and macro-F1 by 0.00465.
No held-out standardized input values exceed the calibrated INT8 input range.
The linear baseline has higher accuracy than the MLP; the MLP has slightly higher
macro-F1. These are descriptive results from one correlated synthetic sequence,
not a statistically established advantage. Stationary-class recall is weak:
5.79% for the floating-point MLP and 4.96% for INT8. Further experiments should
address class behavior using validation data and a fresh final test interval.

## Verification

- All 25 Python and 10 RTL regression test methods pass.
- All 592 held-out examples match the Python integer model through the hidden
  layer, output layer, and connected RTL MLP, with and without seeded stalls.
- Raw CSV → features → saved normalization → integer inference matches every
  exported held-out prediction.
- Replay produces 2,997 predictions, including the final snapshot for which no
  future label is available. Replay includes training periods and is not itself
  a held-out accuracy measurement.
- Repeated local training produced identical reports and all artifact hashes.

The measured environment uses Python 3.12.7, NumPy 1.26.4, scikit-learn 1.7.1,
and Icarus Verilog 13.0. Exact package pins are in `requirements-training.txt`.
Floating-point training may differ across platforms; each run exports its own
parameters and requires exact integer/RTL agreement against those parameters.

Raw reports: [dataset](../results/lob-synthetic/data-summary.json),
[training](../results/lob-synthetic/report.json),
[RTL](../results/lob-synthetic/rtl_report.json), and
[replay](../results/lob-synthetic/replay_report.json).
Regenerate them with `make lob-benchmark`. Source data, weights, and predictions
remain under ignored `build/` directories; only small reference reports are
intentionally versioned.

All computation has been verified in software/simulation. Host preprocessing,
clock frequency, synthesis timing/resources, board behavior, and exchange-feed
latency have not been measured as an integrated hardware system.
