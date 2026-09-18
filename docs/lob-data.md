# Order-book data contract

The default source is **synthetic**, generated locally. Its pressure-driven price
and volume rules provide a test signal, not a calibrated exchange simulation or
evidence of market predictability. No market account or hardware is required.

## Create the dataset

```sh
make lob-data
```

This generates 3,000 five-level snapshots under `build/lob-data/`, followed by
`prepared/dataset.json` and `prepared/summary.json`. Seed 2026 reproduces the data.
The metadata identifies the source as synthetic and records its SHA-256 hash.

## Import your own CSV

For each snapshot, supply these columns in this order:

```text
timestamp_ns,session_id,symbol,bid_price_1,bid_size_1,ask_price_1,ask_size_1,...,bid_price_5,bid_size_5,ask_price_5,ask_size_5
```

The ellipsis above represents levels 2–4; it is not a literal CSV column. The
generated CSV is a complete format example. Prices are in quote-currency units,
sizes in consistent quantity units. Timestamps are nonnegative integer
nanoseconds in strictly increasing order. Normalize duplicate event timestamps
upstream while preserving event order. Only one instrument is accepted per file.
Session IDs must be contiguous blocks and identify boundaries across which a
feature window or label should not extend.

Prices and displayed sizes must be positive and finite. Bids strictly descend,
asks strictly ascend, and best bid must be less than best ask. This first importer
requires all configured depth levels; it rejects missing depth, locked/crossed
books, nonfinite values, and malformed rows. It does not reconstruct snapshots
from incremental exchange messages.

```sh
python3 -m scripts.prepare_lob --input /path/to/snapshots.csv \
  --kind real --source "Provider, instrument, dates, and applicable data license" \
  --tick-size 0.01 --output build/my-lob/prepared
```

Use `--levels`, `--lookback`, `--horizon`, and `--threshold-ticks` to change the
explicit dataset specification. Retain provenance; the software never infers
that data is real merely because it came from a CSV. Existing source metadata
must agree with the requested kind, depth, tick size, and file checksum.

## Features and target

At each level, use bid/ask distance from the current mid-price in tick units and
`log1p` of bid/ask size. Add spread in ticks and total displayed-depth imbalance
`(bid_depth - ask_depth) / (bid_depth + ask_depth)`. With five levels this gives
22 values per snapshot. Flatten four snapshots, oldest first: **88 features**.
Features use only observations at or before the decision timestamp.

The label compares `mid[t + 5]` with `mid[t]`. More than +0.5 tick is **up (2)**,
less than -0.5 tick is **down (0)**, and values inside or on the thresholds are
**stationary (1)**. Horizon units are snapshots, not elapsed time. This is a
simple endpoint target, not an implementation of DeepLOB's labeling method.

Raw rows are first divided chronologically into 60% training, 20% validation, and
20% test. Examples are built independently inside each partition, discarding any
whose history or future label leaves that partition or crosses a session. With
one session and default settings the sample counts are 1,792 / 592 / 592.
This deliberately avoids even shared history rows between partitions. Adjacent
examples *within* a partition overlap and are not independent observations.

Fit normalization and quantization scales using training examples only. Use
validation to select model settings and reserve test examples for final reporting.
The leakage precautions follow the general principles in
[scikit-learn's preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
For research context on order-book prediction, see
[DeepLOB](https://arxiv.org/abs/1808.03668); the compact MLP here is a separate baseline.
