# Parallel neuron RTL

The dense layer can compute several output neurons simultaneously. Each lane
has a MAC accumulator and requantizer; all lanes consume the same input feature
each cycle with their own weights. Results are registered, then serialized onto
the existing ready/valid output. Partial final groups are masked. Reset flushes
all lanes and any partly transmitted output vector.

This changes compute scheduling, not the trained function. Weights, biases,
rounding, saturation, output order, and model files retain their existing format.
Defaults remain one lane per layer. Additional lanes exchange arithmetic and
memory-read resources for fewer compute cycles.

## Reproduce without hardware

Generate the model once if it is not already present:

```sh
make improve-lob
```

Then compare the saved model across five lane configurations:

```sh
make improve-rtl
# Equivalent, including on Windows with Python and Icarus on PATH:
python -m scripts.benchmark_rtl
```

Reports go to `build/rtl-parallel/`. The summary identifies model artifacts by
SHA-256 and only changes its status to `passed` after all configurations pass.
Each configuration tests both individual layers and the full MLP on every saved
held-out example, with and without source gaps and receiver stalls. The benchmark
checks the actual saved ROM files. Training and source-data preparation are not
repeated by this command.

For a single configuration or a different model:

```sh
python -m scripts.verify_export --model build/lob-improved/model --hidden-lanes 4 --output-lanes 3
python -m scripts.benchmark_rtl --model build/digits --lanes 4:3 --output build/digit-rtl-parallel
```

Nondefault verification uses a separate report filename, such as
`rtl_report_4x3.json`. Custom benchmark lane pairs always include the 1:1 baseline.
Lane counts cannot exceed the layer's output count. In an RTL instantiation, set
`mlp.HIDDEN_LANES` and `mlp.OUTPUT_LANES`, or `dense_layer.LANES` directly.

## Recorded LOB measurements

Same **88→16→3** model, seed 2028, and all **592** saved test examples:

| Hidden lanes | Output lanes | Logical MAC instances | Final input → final output cycles | Input vector start interval |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2 | 1,513 | 1,544 |
| 2 | 1 | 3 | 793 | 824 |
| 4 | 1 | 5 | 433 | 464 |
| 4 | 3 | 7 | 397 | 464 |
| 8 | 3 | 11 | 217 | 284 |

These are measured clock-period differences with a continuously available input
and always-ready final receiver. Minimum and maximum values agree across the
592 examples for these configurations. Source gaps and receiver stalls have
separate reports and increase measured intervals.

Four hidden lanes plus three output lanes reduce compute latency by **73.76%**
(a **3.81×** cycle-count ratio) while producing identical logits. Four hidden
lanes with one output lane use fewer MAC instances and achieve the same input
start interval, with a slightly longer final-output latency. Eight hidden lanes
further reduce cycles at additional resource cost. Select a configuration after
considering both latency and resource use; the defaults retain the small design.

The model's synthetic-test accuracy remains **51.86%**. Parallel scheduling does
not improve prediction quality. The 64→16→10 digit model also passes all 450 saved
test examples with four hidden and three output lanes.

Reference reports are in [results/rtl-parallel](../results/rtl-parallel/report.json),
with the digit verification in [digits-4x3.json](../results/rtl-parallel/digits-4x3.json).

## Verification and limits

Regression tests sweep lane counts, non-divisible output groups, signed extrema,
nonzero biases, ReLU, shifts, stalls, repeated vectors, and reset recovery after
partial output transmission. An intentionally slower output layer checks internal
backpressure. The scoreboard checks exact isolated-vector cycle counts and
records repeated-vector input intervals.

Memory reads remain asynchronous and require multiple independent weight reads
for parallel lanes. This implementation does not establish BRAM inference, DSP
mapping, LUT usage, achieved clock frequency, power, or wall-clock latency.
Synthesis and timing analysis are still needed to choose a physical configuration.
The reported MAC count is the number of logical RTL instances, not a vendor
resource report. See [the interface and cycle formulas](interfaces.md).
