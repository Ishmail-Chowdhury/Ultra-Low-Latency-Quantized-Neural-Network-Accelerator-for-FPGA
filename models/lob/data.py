"""Strict snapshot CSV ingestion and split-local, causal LOB examples."""

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Sequence

CLASSES = ["down", "stationary", "up"]


@dataclass(frozen=True)
class Snapshot:
    timestamp_ns: int
    session_id: str
    symbol: str
    bid_prices: tuple[float, ...]
    bid_sizes: tuple[float, ...]
    ask_prices: tuple[float, ...]
    ask_sizes: tuple[float, ...]

    @property
    def mid(self) -> float:
        return (self.bid_prices[0] + self.ask_prices[0]) / 2


@dataclass(frozen=True)
class FeatureConfig:
    levels: int = 5
    lookback: int = 4
    horizon: int = 5
    tick_size: float = 0.01
    threshold_ticks: float = 0.5
    include_mid_changes: bool = False

    def __post_init__(self):
        for name in ("levels", "lookback", "horizon"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.tick_size) or self.tick_size <= 0:
            raise ValueError("tick_size must be positive and finite")
        if not math.isfinite(self.threshold_ticks) or self.threshold_ticks < 0:
            raise ValueError("threshold_ticks must be finite and nonnegative")
        if type(self.include_mid_changes) is not bool:
            raise ValueError("include_mid_changes must be boolean")


def columns(levels: int) -> list[str]:
    return ["timestamp_ns", "session_id", "symbol"] + [
        f"{field}_{level}" for level in range(1, levels + 1)
        for field in ("bid_price", "bid_size", "ask_price", "ask_size")]


def read_snapshots(path: Path, levels: int = 5) -> list[Snapshot]:
    if type(levels) is not int or levels < 1:
        raise ValueError("levels must be positive")
    result = []
    seen_sessions = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns(levels):
            raise ValueError(f"CSV header must be exactly: {','.join(columns(levels))}")
        for line, row in enumerate(reader, start=2):
            try:
                if None in row or any(v is None or v.strip() == "" for v in row.values()):
                    raise ValueError("missing or extra fields")
                timestamp = int(row["timestamp_ns"])
                values = [[float(row[f"{name}_{i}"]) for i in range(1, levels + 1)]
                          for name in ("bid_price", "bid_size", "ask_price", "ask_size")]
                if timestamp < 0 or any(not math.isfinite(x) or x <= 0 for group in values for x in group):
                    raise ValueError("timestamp must be nonnegative; prices/sizes must be positive and finite")
                bids, bid_sizes, asks, ask_sizes = values
                if bids[0] >= asks[0]:
                    raise ValueError("locked or crossed book")
                if any(a <= b for a, b in zip(bids, bids[1:])) or any(a >= b for a, b in zip(asks, asks[1:])):
                    raise ValueError("price levels must strictly descend on bids and ascend on asks")
                session, symbol = row["session_id"], row["symbol"]
                if result:
                    if timestamp <= result[-1].timestamp_ns:
                        raise ValueError("timestamps must strictly increase; normalize duplicate event times upstream")
                    if symbol != result[-1].symbol:
                        raise ValueError("one instrument per file is required")
                    if session != result[-1].session_id and session in seen_sessions:
                        raise ValueError("session IDs must form contiguous blocks")
                seen_sessions.add(session)
                result.append(Snapshot(timestamp, session, symbol, tuple(bids), tuple(bid_sizes),
                                       tuple(asks), tuple(ask_sizes)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid CSV row {line}: {exc}") from exc
    if not result:
        raise ValueError("snapshot file is empty")
    return result


def feature_names(config: FeatureConfig) -> list[str]:
    per_row = [f"{name}_{level}" for level in range(1, config.levels + 1)
               for name in ("bid_distance_ticks", "ask_distance_ticks", "log_bid_size", "log_ask_size")]
    per_row += ["spread_ticks", "depth_imbalance"]
    names = [f"lag_{lag}/{name}" for lag in range(config.lookback - 1, -1, -1) for name in per_row]
    if config.include_mid_changes:
        names += [f"mid_change_ticks/lag_{lag}_to_{lag - 1}" for lag in range(config.lookback - 1, 0, -1)]
    return names


def snapshot_features(row: Snapshot, config: FeatureConfig) -> list[float]:
    if len(row.bid_prices) != config.levels:
        raise ValueError("snapshot depth does not match feature configuration")
    values = []
    for bid, ask, bid_size, ask_size in zip(row.bid_prices, row.ask_prices, row.bid_sizes, row.ask_sizes):
        values.extend([(row.mid - bid) / config.tick_size, (ask - row.mid) / config.tick_size,
                       math.log1p(bid_size), math.log1p(ask_size)])
    bid_depth, ask_depth = sum(row.bid_sizes), sum(row.ask_sizes)
    values += [(row.ask_prices[0] - row.bid_prices[0]) / config.tick_size,
               (bid_depth - ask_depth) / (bid_depth + ask_depth)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("feature arithmetic exceeded finite range")
    return values


def feature_window(rows: Sequence[Snapshot], decision: int, config: FeatureConfig) -> list[float]:
    start = decision - config.lookback + 1
    if start < 0 or decision >= len(rows):
        raise ValueError("insufficient feature history")
    window = rows[start:decision + 1]
    if any(row.session_id != window[0].session_id for row in window):
        raise ValueError("feature window crosses a session")
    values = [value for row in window for value in snapshot_features(row, config)]
    if config.include_mid_changes:
        values += [(b.mid - a.mid) / config.tick_size for a, b in zip(window, window[1:])]
    return values


def movement_label(current_mid: float, future_mid: float, config: FeatureConfig) -> int:
    delta = (future_mid - current_mid) / config.tick_size
    # Treat machine-precision noise at the threshold as an exact tie (stationary).
    if math.isclose(abs(delta), config.threshold_ticks, rel_tol=1e-9, abs_tol=1e-9):
        return 1
    return 2 if delta > config.threshold_ticks else 0 if delta < -config.threshold_ticks else 1


def examples_in_range(rows: Sequence[Snapshot], config: FeatureConfig, start: int, stop: int) -> dict:
    """Construct examples whose complete feature/label support stays in [start, stop)."""
    if not 0 <= start < stop <= len(rows):
        raise ValueError("invalid raw-row interval")
    examples = []
    for decision in range(start + config.lookback - 1, stop - config.horizon):
        first = decision - config.lookback + 1
        end = decision + config.horizon
        if any(row.session_id != rows[decision].session_id for row in rows[first:end + 1]):
            continue
        examples.append({"feature_start": first, "decision_index": decision, "label_end": end,
                         "timestamp_ns": rows[decision].timestamp_ns,
                         "features": feature_window(rows, decision, config),
                         "label": movement_label(rows[decision].mid, rows[end].mid, config)})
    if not examples:
        raise ValueError("interval has no usable examples; provide more rows or shorter windows")
    return {"raw_start": start, "raw_stop": stop, "examples": examples}


def prepare(rows: Sequence[Snapshot], config: FeatureConfig) -> dict:
    """Split raw rows 60/20/20; build windows and future labels inside each split."""
    boundaries = [0, int(len(rows) * 0.6), int(len(rows) * 0.8), len(rows)]
    return {name: examples_in_range(rows, config, start, stop)
            for name, start, stop in zip(("train", "validation", "test"), boundaries, boundaries[1:])}


@dataclass(frozen=True)
class Standardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]):
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("standardizer requires a nonempty rectangular matrix")
        if any(not math.isfinite(v) for row in rows for v in row):
            raise ValueError("standardizer requires finite data")
        mean = tuple(math.fsum(column) / len(rows) for column in zip(*rows))
        scale = tuple(math.sqrt(math.fsum((v - m) ** 2 for v in column) / len(rows))
                      for m, column in zip(mean, zip(*rows)))
        return cls(mean, tuple(s if s > 1e-12 else 1.0 for s in scale))

    def transform(self, values: Sequence[float]) -> list[float]:
        if len(values) != len(self.mean) or len(self.scale) != len(self.mean):
            raise ValueError("standardizer dimensions differ")
        if any(not math.isfinite(s) or s <= 0 for s in self.scale):
            raise ValueError("standardizer scales must be finite and positive")
        transformed = [(v - m) / s for v, m, s in zip(values, self.mean, self.scale)]
        if any(not math.isfinite(v) for v in transformed):
            raise ValueError("nonfinite standardized feature")
        return transformed
