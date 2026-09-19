# Real market-data smoke test

The Nasdaq sample now runs through book reconstruction, feature extraction,
model evaluation, INT8 export, source replay, and parallel RTL simulation.
The current models do **not** beat the stationary baseline on accuracy in this
small sample. Functional RTL agreement is successful; predictive usefulness is
not established.

## Data actually tested

The first **2,000,000 complete BinaryFILE messages** from Nasdaq's public
[`01302019.NASDAQ_ITCH50.gz`](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/)
produce **990 five-level AAPL snapshots** on 2019-01-30, approximately
04:00:03–05:27:34 exchange-local time. This is a deliberately bounded premarket
prefix, not a full trading day or a representative regular-hours benchmark.
The downloaded uncompressed prefix is 58,039,979 bytes. Raw data remains under
ignored `build/real-data/`; only reports and checksums are versioned.

The downloader starts at the beginning of the gzip archive and stops at a complete
message boundary. It records a SHA-256 of the retained prefix. It does not fetch
or verify the complete archive's gzip trailer or provider checksum. The initial
500,000-message access probe yielded 309 valid snapshots; the expanded prefix
was selected for sample size before evaluating either model. No symbol, label,
or model settings were changed in response to the final scores.

## Results

Two different protocols were run. Their scores use different test intervals and
must not be treated as a paired model comparison.

| Experiment | Scored examples | INT8 accuracy | INT8 macro-F1 | Stationary baseline accuracy |
| --- | ---: | ---: | ---: | ---: |
| Existing synthetic-trained model, frozen; entire new sample | 982 | 37.88% | 0.2892 | 42.36% |
| Real-data training and chronological model selection; final 20% reserved | 190 | 27.37% | 0.2464 | 29.47% |

The frozen model used its original normalization, scales, weights, and thresholds.
**47.61%** of scaled input values lay outside its representable INT8 input range,
before rounding and clipping. This is evidence of a substantial input-distribution
mismatch. The score includes the boundary-supported windows throughout the new
sample; no fitting uses these data in the frozen experiment.

The separate real-data experiment uses the existing three expanding validation
folds and six predeclared baseline candidates. It selected the balanced
**88→8→3** MLP, then refitted on **784 development examples**, leaving **190**
test examples. The float model scored 27.89% accuracy and 0.2477 macro-F1;
the balanced logistic comparator scored 27.89% and 0.2432. The stationary baseline
macro-F1 was 0.1518. All three classes occur in the final test: 45 down,
56 stationary, and 89 up. See the full confusion matrices in the reports.

All **982 frozen-model examples** match the saved-ROM parallel MLP in RTL, with
and without stalls. All **190 real-trained holdout examples** match both layers
individually and the complete parallel MLP, with and without stalls. Raw-source
replay also reproduces every held-out prediction. Four hidden and three output
lanes were used. No board, CUDA, or FPGA synthesis was involved.

These are small, overlapping windows from one early-morning period and one
instrument. They validate integration and expose a generalization problem;
they do not establish trading performance. Subsequent model selection needs
new, larger development periods and untouched final test days.

## Reproduce

Install the documented training dependencies and Icarus. If the frozen synthetic
model does not exist, generate it once with `make improve-lob`. Then:

```sh
make fetch-nasdaq
make test-nasdaq
```

The first command explicitly uses the network. The second works from the saved
prefix and runs the full evaluation chain. On Windows without Make, run the
individual `python -m ...` commands from the `test-nasdaq` target in the Makefile,
using your virtual environment's Python. For example:

```sh
python -m scripts.fetch_itch_sample --messages 2000000 --output build/real-data/nasdaq/prefix-2m.itch
python -m scripts.import_itch --input build/real-data/nasdaq/prefix-2m.itch --output build/real-data/nasdaq/aapl-2m.csv --symbol AAPL --date 2019-01-30 --source "Nasdaq 2019-01-30 public ITCH prefix, 2000000 messages"
python -m scripts.evaluate_external_lob --input build/real-data/nasdaq/aapl-2m.csv --output build/real-data/nasdaq/frozen-evaluation.json --rtl
```

The external evaluator checks the model manifest and source sidecar, preserves
the trained preprocessing, and rejects a source whose hash equals the model's
training-source hash. A different hash alone does not prove disjointness: the
caller must supply genuinely new observations.

## Import behavior

The ITCH adapter requires a Start of Messages event and the selected stock's
directory entry. It reconstructs displayed orders using add, attributed add,
execution, execution-with-price, cancel, delete, and replacement messages.
Executions reduce the resting order's quantity at its original display price;
trade prints do not alter displayed depth. Unknown message types, truncated
frames, missing order references, invalid quantities, and reversed target-symbol
timestamps fail conversion. It supports the documented ITCH 5.0 message set,
not arbitrary future feed extensions or packet captures with other framing.
See Nasdaq's [ITCH specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification_5.0.pdf)
and [BinaryFILE framing](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/binaryfile.pdf).

Snapshots are emitted after target-symbol displayed-book changes, including
changes below the selected depth. Missing depth, halts, and locked/crossed books
are excluded and start a new session when valid data resumes, preventing windows
or labels from bridging the excluded interval. Prices are divided by 10,000;
sizes are shares. The selected AAPL task uses a $0.01 tick. Timestamp ties are
incremented by one nanosecond to preserve source order; 67 rows needed adjustment
in this sample. Timestamps denote exchange-local time since midnight, not UTC
epoch timestamps. The supplied date must match the source; ITCH messages do not
encode it. Each conversion handles one day and one instrument.

The paired LOBSTER CSV adapter verifies row alignment, decimal timestamps,
price/size columns, and ordering. It converts integer price units, normalizes
duplicate times, and breaks sessions around missing depth, auction markers, and
halts. This adapter has only been validated with unit fixtures because the
provider's sample downloads were unavailable during this run. For local files:

```sh
python -m scripts.import_lobster --messages /path/to/message.csv --orderbook /path/to/orderbook.csv --symbol AAPL --date 2012-06-21 --source "LOBSTER sample, provider and date" --output build/real-data/lobster/aapl.csv
```

Select the matching message/orderbook pair and its actual date/symbol. The
result works with the external evaluator or existing chronological trainer.

## Other supplied sources

Access was checked on 2026-09-18. These sources have **not** been tested with the
model; source discovery or fixture tests do not count as dataset evaluation.

| Source | Observed result | Needed to complete testing |
| --- | --- | --- |
| [LOBSTER samples](https://lobsterdata.com/info/DataSamples.php) | Old archive URLs returned HTML; legacy data hosts did not resolve here | Downloaded matching message/orderbook CSV pair or a working provider archive link |
| [FI-2010](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) | Official metadata/file listing retrieved; Metax download service returned 405 disabled; Etsin UI download endpoint returned 500 | Official `BenchmarkDatasets.zip` or restored download service, then a separate prepared-feature benchmark adapter |
| [Coinbase full channel](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels#full-channel) | Live server rejected the subscription because authentication is required | Locally configured feed access or a saved level-3 snapshot plus contiguous full-channel capture |

FI-2010 is distributed as 144 normalized feature rows and five label rows, with
labels 1=up, 2=stationary, 3=down and predefined anchored folds. Those values
cannot be passed into this project's raw-price importer. Its official metadata
attributes the CC BY 4.0 dataset to Adamantios Ntakaris, Martin Magris, Juho
Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis. The archive is
1,864,361,899 bytes; its provider SHA-256 is recorded in source-status.json.

Coinbase's full feed needs a synchronized level-3 snapshot and sequenced updates;
a documentation example is insufficient to reconstruct a live book. No
credentials were requested or used. The live rejection is recorded rather than
assuming the public-feed overview guarantees access to this particular channel.

## Recorded artifacts

The subsequent [complete-day data preparation milestone](full-day-data.md)
records full regular-hours downloads separately from the prefix results below.

Under [results/nasdaq-sample](../results/nasdaq-sample/source-status.json):

- [Access status for all four sources](../results/nasdaq-sample/source-status.json)
- [Prefix download metadata](../results/nasdaq-sample/download.json) and [conversion audit](../results/nasdaq-sample/import.json)
- [Frozen model metrics and RTL](../results/nasdaq-sample/frozen-evaluation.json)
- [Real-data training/selection](../results/nasdaq-sample/real-training.json)
- [Held-out source replay](../results/nasdaq-sample/real-replay.json) and [parallel RTL](../results/nasdaq-sample/real-rtl.json)

The ordinary regression remains offline. It now includes malformed market-data
frames, order lifecycles, book aggregation, non-book trade prints, halt/session
handling, aligned LOBSTER rows, source metadata corruption, and frozen-model
evaluation without modifying model artifacts.
