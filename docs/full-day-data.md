# Complete-day Nasdaq data preparation

The public [Nasdaq ITCH archive](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/)
provides complete trading days without credentials. The downloader consumes each
entire gzip archive, verifies its trailer and HTTP byte count, records SHA-256
hashes, and retains AAPL messages plus daily system events. Raw data stays under
the ignored `build/` directory.

## Verified milestone

The 2019-07-30, 2019-08-30, 2019-10-18, and 2019-12-30 archives have been
downloaded and validated.
Their provenance and regular-hours reconstruction audits are saved under
[results/nasdaq-days](../results/nasdaq-days). Each produces 23,400 valid
five-level books on a one-second grid from 09:30:00 inclusive to 16:00:00
exclusive, in exchange-local time.

These provider archives omit the BinaryFILE zero-length terminator. The audit
records that explicitly; completeness checks use the entire HTTP payload, gzip
integrity, and the ordered daily system events O, S, Q, M, E, C.

The four-day CSV contains 93,600 snapshots with 100% grid coverage on each day.
Its [provenance and split metadata](../results/nasdaq-days/snapshots.metadata.json)
are recorded separately from trained-model results. The dates were selected
before model evaluation: July 30 and August 30 for training, October 18 for
validation, and December 30 for testing, all in 2019. These dates are nonconsecutive, so
results will span different market periods rather than adjacent sessions.

## Resume and prepare

From the repository root:

```sh
mkdir -p build
cc -O3 -Wall -Wextra -Werror scripts/itch_filter.c -o build/itch-filter
python3 -m scripts.fetch_itch_days --scanner build/itch-filter --ranged-download
python3 -m scripts.prepare_itch_days
```

The native scanner is optional; omit `--scanner build/itch-filter` to use Python.
Ranged downloads use four connections per archive, with at most two archives
concurrently. Rerunning reuses completed retained files after checking their
hashes and resumes completed archive parts after checking source identity.
Allow roughly 15 GB of network transfer for a fresh four-day download and
additional disk space for temporary archive assembly.

Preparation requires at least four complete days and at least 99% regular-hours
coverage for each. Every grid point uses only messages at or before its timestamp;
same-timestamp updates are applied together. Invalid books and trading halts
break sessions so feature and label windows cannot bridge them. Timestamps are
calendar-day offsets plus exchange-local nanoseconds, **not UTC epoch times**.
The CSV sidecar records these choices, source hashes, coverage, and day splits.

`models/lob/day_split.py` enforces chronological, disjoint whole-day partitions.
The optional `include_mid_changes` feature adds causal midpoint differences;
existing feature configuration remains unchanged by default.

The original ingestion milestone passed 67 Python tests, including scanner agreement, malformed and
incomplete streams, causal grid sampling, session boundaries, day partitions,
and causal midpoint features. Follow the separate
[prediction-quality study](prediction-quality.md) for model evaluation.
