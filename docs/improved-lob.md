# INT8-aware LOB model improvement

```sh
make improve-lob
```

This uses a fresh synthetic sequence, seed **2028**, and the existing three-fold
chronological protocol. Both the selected model and the prior 8-neuron balanced
design are refitted on the same development data and tested on the same reserved
592 examples. Artifacts go to `build/lob-improved/`; earlier runs are preserved.

## Changes

MLP selection scores **quantized integer inference** on validation data. Each fold
calibrates scales only from its own training prefix. The predeclared candidates
combine hidden sizes 8/16, L2 strengths 1/5, and stationary-logit offsets
0/0.125/0.25/0.5/0.75: 20 MLP configurations, all with balanced training weights.
Each of four fitted networks per fold is reused for five offsets; model copies
prevent offsets from accumulating. A balanced logistic comparator is evaluated
separately using floating-point predictions. Mean validation macro-F1 selects
the winner within each family, with declared order breaking exact ties.

The offset is added to the existing stationary output bias before quantization.
It requires no additional RTL operation. Exported weights/biases still pass the
accumulator-bound checks. Selection is written before final-test examples are
constructed; no hyperparameters or scales are selected using final-test labels.

The winner is **88→16→3**, balanced training, `alpha=5.0`, and a **+0.125 stationary
logit offset**. Mean INT8 validation macro-F1 is 0.5214, versus 0.4927 for the
previous 8-neuron balanced design on the same validation folds.

## Paired final-test comparison

Both designs are refitted on the same 2,392 development examples using seed 2028.
The baseline uses 8 hidden neurons, `alpha=1.0`, balanced training, and no offset.

| INT8 metric | Previous design, retrained | Improved design |
| --- | ---: | ---: |
| Accuracy | 51.18% (303/592) | 51.86% (307/592) |
| Macro-F1 | 0.4545 | 0.4744 |
| Stationary recall | 15.70% (19/121) | 22.31% (27/121) |
| Hidden neurons | 8 | 16 |

Stationary recall rises **6.61 percentage points**, accuracy rises 0.68 points,
and macro-F1 rises 0.0199. This is a modest descriptive improvement on one
correlated synthetic sequence, not an established real-market or statistically
significant advantage. Most stationary examples are still missed. The earlier
60.81% result used seed 2027 and a different test set, so it is not the comparator.

The selected model takes 1,513 no-stall cycles from final input transfer to final
output transfer in simulation; the prior architecture's controller formula gives
761 cycles. Weight entries grow from 728 to 1,456, while two MAC units remain,
one per layer. Better classification costs more sequential computation and
storage. No achieved hardware clock or wall-clock latency is claimed.

## Verification

All 592 test examples match the improved model through both RTL layers and the
connected MLP, with and without stalls. Source replay checks labels, splits,
normalization, inputs, intermediate values, and final scores. It independently
recomputes the saved baseline predictions and every reported comparison delta.
The baseline checksum entry is required; omitting it cannot skip the audit.
Corruption tests reject incorrect baseline scores and inflated gains.

Source data, selected and baseline models, and predictions remain under ignored
`build/` directories. Versioned reference reports:
[training/selection](../results/lob-improved/report.json),
[RTL](../results/lob-improved/rtl_report.json), and
[replay/comparison audit](../results/lob-improved/replay_report.json).

`make walk-forward` retains the earlier procedure. Use `--profile stationary` on
the training CLI to apply the improvement procedure to a properly licensed real
CSV. Reserve new final data for future tuning; this test interval has now been
examined. The same pinned Python dependencies and simulator apply.
