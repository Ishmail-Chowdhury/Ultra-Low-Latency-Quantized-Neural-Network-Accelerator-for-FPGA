# First trained benchmark

The initial model is a **64 → 16 → 10** MLP with ReLU on the hidden layer and
unrectified output logits. Training uses scikit-learn's CPU `MLPClassifier` with
L-BFGS. This keeps the first small model independent of a GPU or PyTorch setup;
the explicit weight/bias export and integer inference remain framework-neutral.
PyTorch training or quantization-aware training can be added later.

## Dataset and split

Use the 1,797 8×8 digit images bundled with `sklearn.datasets.load_digits`.
Pixel features lie in `[0, 16]`; divide by 16 before floating-point training.
The split is stratified, 75% training / 25% testing, seed 2026, giving 1,347
training samples and 450 held-out samples. Split indices are exported.

This is a sample-level split of a small bundled benchmark, not a writer-disjoint
evaluation. It establishes a working training/export/hardware path; it provides
no evidence about market prediction or a production classifier.

The data originates from E. Alpaydin and C. Kaynak's **Optical Recognition of
Handwritten Digits** dataset, UCI Machine Learning Repository,
[DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).
UCI lists it under [CC BY 4.0](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
The loader's bundled subset is described in the
[scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
The source dataset is not copied into this repository. Locally generated test
vectors are transformed versions of its pixel features; retain attribution if
redistributing those vectors.

## Training and calibration

The training command fixes the seed and limits numerical-library threads to one.
It uses a 16-neuron hidden layer, `alpha=0.01`, L-BFGS, at most 1,000 iterations,
and 50,000 function evaluations. A convergence warning fails the run.
L-BFGS is suitable for small datasets per the
[MLPClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

- Input scale is `2^-6`; all normalized dataset pixel values encode exactly.
- Weight scale is the smallest power of two covering the layer's maximum absolute
  weight with the positive INT8 range.
- Hidden/output scales cover maximum absolute **training-set** activations/logits.
- Bias scale is the product of input and weight scales.
- Each layer's integer shift follows directly from these scale exponents.
- Rounding and saturation follow [the arithmetic contract](arithmetic.md).
- Both floating-point and INT8 classification use the first maximum index on ties.

Test data is used only for evaluation and exported RTL regression vectors. It is
not used for fitting weights or selecting calibration scales. No hyperparameter
search is performed on this split. The initial acceptance gate requires at least
90% floating-point accuracy and an INT8 accuracy drop of at most 2 percentage
points; this is an engineering threshold for the baseline, not a statistical
generalization guarantee.

## Reproduce

For a fresh environment, use Python 3.11 or 3.12 and install the pinned training
dependencies. The core and RTL tests do not need these Python packages.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
make test
make benchmark
```

An Icarus installation is also required; see the main README. `make benchmark`
trains, exports, and verifies every held-out vector through each layer separately
and through the connected RTL MLP, with and without stalls. The test suite also
uses a slow second layer to exercise internal backpressure and reset recovery.

Outputs under `build/digits/` include:

- `network.json`: topology, scale exponents, classes, preprocessing, and seed.
- `layer0/`, `layer1/`: integer models and hexadecimal hardware memories.
- `float_parameters.json`: row-major floating-point weights and biases.
- `split.json`: exact training/test indices.
- `vectors.json`: all 450 test inputs, intermediate accumulators, and outputs.
- `report.json`: accuracy, layer errors, acceptance result, versions, and SHA-256 hashes.
- `rtl_report.json`: bit-exact checks and cycle/stall measurements.

Generated files remain outside version control. Small reference reports are
intentionally retained under `results/digits/`. Repeated local training with the
pinned environment produced identical artifact hashes; floating-point training
can differ across BLAS implementations and platforms. Each run records its own
hashes, and integer RTL comparison must be exact against that run's exported model.

Training dependencies are externally installed NumPy, SciPy, scikit-learn,
joblib, and threadpoolctl (BSD-family licenses). Icarus is an external simulator;
its source is not vendored. No third-party implementation code is copied here.
