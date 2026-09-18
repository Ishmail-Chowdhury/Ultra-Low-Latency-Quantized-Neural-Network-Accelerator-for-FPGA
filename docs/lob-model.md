# Software-only LOB model and replay

Run the complete pipeline without FPGA hardware:

```sh
make lob-benchmark
```

Python training dependencies are the existing `requirements-training.txt` pins.
Icarus is required only for the RTL simulation stage. To run just data, training,
and software predictions:

```sh
make lob-data
make train-lob
make predict-lob
```

The commands write a generated CSV under `build/lob-data/` and model artifacts
under `build/lob/`. `predictions.csv` contains 2,997 causal predictions from the
3,000-row default file, after the initial three history rows. This file includes
training and validation periods; use the 592 held-out examples in `report.json`
and `vectors.json` for evaluation, not accuracy over the entire replay file.

## Model selection and quantization

The [data specification](lob-data.md) defines 88 inputs and three outputs in the
order down, stationary, up. Standardization uses only training means and
population standard deviations. Constant features use scale one. Compare hidden
widths 8 and 16 using validation macro-F1, choosing the smaller on exact ties.
Neither test labels nor test statistics select the model.

Each candidate is a ReLU MLP trained with Adam, batch size 128, learning rate
0.001, L2 `alpha=1.0`, a fixed seed, and no training shuffle. Training stops after
20 epochs without sufficient training-loss improvement or at 1,000 epochs.
Scikit-learn's random validation-based early stopping is disabled. A convergence
warning fails training. Full selected hyperparameters and candidate validation
metrics are exported. The initial selected model is **88→16→3**.

After selection, training examples alone calibrate symmetric power-of-two input,
weight, hidden, and output scales. Biases use input×weight scales. The shared INT8
reference implements round-to-nearest with ties away from zero and saturation.
The final logits are signed integers; class selection uses argmax, with the first
class winning ties. Logits are scores, not calibrated probabilities.

Alongside the MLP, report a training-majority classifier, an always-stationary
classifier, and fixed-parameter logistic regression. Report test accuracy,
macro-F1, per-class recall, class support, and confusion matrices for each.
Macro-F1 always includes all three classes and assigns zero when a class has
neither observations nor predictions.

## Use real snapshots later

First import them using the documented CSV contract, then train and replay:

```sh
python3 -m models.training.train_lob \
  --dataset build/my-lob/prepared/dataset.json --output build/my-lob/model
python3 -m scripts.predict_lob --model build/my-lob/model \
  --input /path/to/snapshots.csv --output build/my-lob/predictions.csv
python3 -m scripts.verify_export --model build/my-lob/model
python3 -m scripts.verify_lob_replay --model build/my-lob/model \
  --input /path/to/snapshots.csv
```

`predict_lob` needs only Python's standard library once a model exists. It loads
saved preprocessing and integer weights, checks artifact hashes, and derives each
prediction only from that snapshot's history. Future labels are not needed: it
can classify the final row of a file. This is an offline causal replay tool; it
loads the CSV into memory and does not implement a live exchange feed or measure
real-time service latency.

The replay verifier compares raw CSV → features → normalization → integer MLP
against every exported test vector. The separate RTL verifier compares each
layer and the full network under continuous traffic and seeded backpressure.
Generated artifacts include model scales, weights, biases, preprocessing, exact
split support indices, source hashes, float parameters, and golden vectors.

## Next software experiments

Use real, properly licensed snapshots and retain dataset provenance. For a new
experiment, add expanding-window validation over training/validation history,
then reserve a fresh final test interval. Test class balancing and regularization
using validation data; stationary-class recall is weak in the initial run. Keep
logistic regression as a comparator. Test input drift, missing-depth handling,
and alternative horizons before attempting performance claims.

Feature extraction and normalization currently run on the host. RTL latency
covers only the dense network on already prepared INT8 inputs. Converting the
feature frontend to fixed point is a separate future software/RTL milestone.
