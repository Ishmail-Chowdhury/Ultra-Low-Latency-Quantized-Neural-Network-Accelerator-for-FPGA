# Initial simulation results

Measured on 2026-09-18 using Python 3.12.7, Icarus Verilog 13.0, NumPy 1.26.4,
SciPy 1.16.0, and scikit-learn 1.7.1. Seed: 2026. The full model is 64→16→10,
signed INT8 weights/activations, signed INT32 accumulation, one MAC per layer
(two MAC units total). Hidden/output shifts are 8 and 6.

| Metric | Result |
| --- | --- |
| Training / held-out samples | 1,347 / 450 |
| Floating-point test accuracy | 96.2222% (433/450) |
| INT8 test accuracy | 95.3333% (429/450) |
| Accuracy drop | 0.8889 percentage points |
| RTL hidden outputs checked per traffic mode | 7,200, all exact |
| RTL output-layer logits checked per traffic mode | 4,500, all exact |
| Connected MLP logits checked per traffic mode | 4,500, all exact |
| Traffic modes | Continuous input / ready output; seeded gaps and stalls |
| Core regression | 13 Python tests and 10 RTL test methods passed |
| Multiplication coverage | All 65,536 INT8 operand pairs |
| Repeated training | Identical report and exported-file hashes on this machine |

## Cycles

All values below measure **accepted final input → accepted final output**, as
edge-to-edge clock differences. These are compute-interface measurements; no DMA,
host transfer, external-memory, or Ethernet time is included. An always-ready
final receiver does not eliminate stalls between the MLP's internal layers.

| Configuration | Observed minimum | Observed maximum |
| --- | ---: | ---: |
| Synthetic 8→4 layer, no stalls | 44 | 44 |
| Trained 64→16 layer, no stalls | 1,072 | 1,072 |
| Trained 16→10 layer, no stalls | 190 | 190 |
| Connected MLP, continuous input and ready receiver | 1,262 | 1,322 |
| Connected MLP, seeded source gaps and output stalls | 1,263 | 1,323 |

The first isolated MLP vector takes 1,262 periods. Later vectors can encounter
internal backpressure, explaining the wider range. Input gaps change pipeline
overlap, so they can reduce internal waiting when latency is measured from the
**last** input; this does not imply better end-to-end throughput. The measured
stall ranges describe this test traffic only, not an upper bound for arbitrary
backpressure.

The JSON reports' unprefixed latency fields describe the final vector in each
run. Their `min_...` and `max_...` fields cover every vector in that run.
`stall_cycles` counts blocked valid output cycles at the measured module's
external output; it does not count internal link stalls in the connected MLP.

## Limits and next experiment

There is no established FPGA clock frequency, timing closure, resource report,
power estimate, hardware latency, or CPU/FPGA speedup. The testbench's clock is a
simulation stimulus and does not demonstrate that the circuit meets that period.
The current memory ports are asynchronous, so BRAM inference is not claimed.

Next, add a Vivado build for the KR260 with an explicit clock constraint and a
BRAM/test-vector harness. Inspect synthesis and timing reports before changing
parallelism or claiming a latency in nanoseconds. Runtime weight loading, AXI,
DMA, Linux, additional precisions, and limit-order-book data remain later work.

Raw reference reports: [training](../results/digits/report.json) and
[RTL](../results/digits/rtl_report.json). Regenerate current reports with
`make benchmark`; see [the benchmark methodology](benchmark.md).
