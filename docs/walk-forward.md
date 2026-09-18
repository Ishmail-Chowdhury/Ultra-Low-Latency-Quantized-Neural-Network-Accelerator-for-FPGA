# Expanding-window model selection without hardware

```sh
make walk-forward
```

This creates a separate 3,000-row synthetic sequence with seed **2027**, runs
chronological model selection, exports the selected integer MLP, verifies source
replay and saved-memory RTL, and writes predictions under
`build/lob-walk-forward/`. The earlier seed-2026 experiment is preserved.

## Protocol fixed before evaluation

The final 20% of raw snapshots is reserved for testing. Three expanding training
prefixes predict successive validation intervals in the preceding data:

| Fold | Training raw rows | Validation raw rows |
| --- | --- | --- |
| 1 | `[0, 1200)` | `[1200, 1600)` |
| 2 | `[0, 1600)` | `[1600, 2000)` |
| 3 | `[0, 2000)` | `[2000, 2400)` |

Rows 2400–2999 are excluded from model selection. Feature history and future
labels must fit completely inside each interval. With four history snapshots
and a five-snapshot label horizon, each 400-row validation interval contributes
392 examples. Training counts are 1,192, 1,592, and 1,992.

Every fold fits its own mean/std normalizer on that fold's training examples.
Class-balanced candidates use a training sample weight of
`n_train / (3 * count_of_its_class)`, so each class has equal total training
weight. Validation metrics remain unweighted. The six predeclared candidates are:

- MLP with 8 hidden neurons, unweighted or balanced.
- MLP with 16 hidden neurons, unweighted or balanced.
- Logistic regression, unweighted or balanced.

The MLP uses Adam and the earlier training settings, with a maximum of 1,500
epochs. Convergence warnings fail the run. The pinned scikit-learn 1.7.1 supports
[MLP sample weights](https://scikit-learn.org/1.7/modules/generated/sklearn.neural_network.MLPClassifier.html).
Expanding prefixes follow the rationale of
[time-series cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split);
the repository constructs examples within raw-row intervals to prevent both
history and future-label overlap across a fold boundary.

Choose the MLP and linear candidate separately by mean validation macro-F1.
Declared candidate order breaks exact ties. `selection.json` is written before
constructing final-test examples. Refit the selected models on the entire
development prefix `[0, 2400)`, refit normalization and INT8 calibration there,
then evaluate the final `[2400, 3000)` interval once per selected model. Final
training has 2,392 examples; final testing has 592.

Earlier validation intervals become training data for later folds and the final
fit. This is intentional expanding-window evaluation. The final test interval
never participates in fitting, class weighting, calibration, or selection.

## Recorded synthetic results

The selected model is **88→8→3 with class-balanced training**. The selected linear
comparator also uses balanced training.

| Final test model | Accuracy | Macro-F1 | Stationary recall |
| --- | ---: | ---: | ---: |
| Floating-point selected MLP | 60.81% | 0.5253 | 16.67% |
| INT8 selected MLP | 60.81% | 0.5209 | 15.00% |
| Selected logistic regression | 58.95% | 0.5214 | 20.00% |
| Training-majority classifier | 43.07% | 0.2007 | 0.00% |

INT8 preserves overall accuracy here, but changes some predictions and lowers
macro-F1 slightly. The linear model has slightly higher macro-F1 than the INT8
MLP and better stationary recall. Fold standard deviations are descriptive;
the folds share training history and do not form independent trials.

The completed run passes all 55 regression test methods. All 592 final held-out
examples match the integer reference through saved-memory RTL with and without
stalls, and raw-CSV replay reproduces the exported predictions exactly.

Do not compare these numbers directly with the earlier seed-2026 test as an
improvement estimate: the data and evaluation protocol changed. Evidence for
balancing comes from its comparisons on the same validation intervals in this
experiment. This remains a toy-data result, not market-performance evidence.

The selected MLP's parameters and golden vectors use the existing export format,
with `evaluation_protocol: walk_forward_v1` and a recorded fold plan. The replay
auditor understands this final development/test split. Saved ROMs are simulated
through both layers and the connected MLP for every held-out sample, with and
without stalls. Host preprocessing remains outside RTL latency measurements.

## Files and real-data use

`build/lob-walk-forward/model/` contains selection scores, per-fold normalization
statistics, split indices, float and integer weights, fitted linear coefficients,
golden vectors, and verification reports. `predictions.csv` in the parent folder
includes development-period predictions; its entire contents are not a held-out
test. Small reference reports are stored in
[results/lob-walk-forward](../results/lob-walk-forward/report.json).

For real snapshots, use the same [CSV contract](lob-data.md) and run:

```sh
python3 -m models.training.walk_forward_lob --input /path/to/book.csv \
  --kind real --source "Provider, symbol, dates, and data license" \
  --output build/real-walk-forward/model
```

Specify tick size, levels, history, and horizon with the documented CLI flags if
they differ from the defaults. Each training prefix must contain all three
classes. Use a fresh final interval for later tuning experiments; the recorded
test interval has now been examined. The next useful extension is to run this
protocol on properly licensed real snapshots or specify the fixed-point feature
frontend, both of which can begin without an FPGA board.
