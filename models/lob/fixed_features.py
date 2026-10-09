"""Integer-only LOB feature arithmetic after decimal ingress and constant export.

Version 1: prices/quantities in units of 1e-4, signed Q24 features, Q30
normalization reciprocals and logarithm table. See docs/fixed-features.md.
"""

import csv
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_HALF_UP, DecimalException
import hashlib
import json
from pathlib import Path

from models.lob.data import FeatureConfig, Standardizer, columns, feature_names
from models.quantization.artifacts import validate_export

PRICE_UNITS = 10000
QUANTITY_UNITS = 10000
FEATURE_BITS = 24
COEFFICIENT_BITS = 30
LOG_BITS = 30
TABLE_BITS = 10
ONE = 1 << FEATURE_BITS
FORMAT = "lob_fixed_q24_v1"


def checked(value, bits, signed=True):
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Value does not fit {'signed' if signed else 'unsigned'} {bits}-bit contract")
    return value


def divide_round(numerator, denominator):
    """Signed 128-bit numerator / positive signed 128-bit divisor; ties away."""
    checked(numerator, 128)
    checked(denominator, 128)
    if denominator <= 0:
        raise ValueError("Division requires a positive denominator")
    quotient, remainder = divmod(abs(numerator), denominator)
    magnitude = quotient + (2 * remainder >= denominator)
    return -magnitude if numerator < 0 else magnitude


def shift_round(value, bits):
    if type(bits) is not int or not 0 <= bits <= 126:
        raise ValueError("Unsupported rounding shift")
    return divide_round(value, 1 << bits)


def decimal_units(value, units):
    """Boundary conversion only: reject sub-unit precision instead of rounding it."""
    try:
        with localcontext() as context:
            context.prec = 80
            number = Decimal(str(value)) * units
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("Value is not exactly representable in the configured decimal units")
            if abs(number) > (1 << 64) - 1:
                raise ValueError("Decimal ingress exceeds 64-bit domain")
            return int(number)
    except (DecimalException, OverflowError) as exc:
        raise ValueError("Invalid decimal input") from exc


@dataclass(frozen=True)
class FixedSnapshot:
    timestamp_ns: int
    session_id: str
    symbol: str
    bid_prices: tuple[int, ...]
    bid_sizes: tuple[int, ...]
    ask_prices: tuple[int, ...]
    ask_sizes: tuple[int, ...]

    def __post_init__(self):
        checked(self.timestamp_ns, 64, False)
        if type(self.session_id) is not str or type(self.symbol) is not str or not self.session_id or not self.symbol:
            raise ValueError("Session and symbol are required")
        depth = len(self.bid_prices)
        if not 1 <= depth <= 32 or any(len(x) != depth for x in (self.ask_prices, self.bid_sizes, self.ask_sizes)):
            raise ValueError("Fixed-point books require matching depths of 1..32")
        for vector, bits in ((self.bid_prices, 32), (self.ask_prices, 32), (self.bid_sizes, 48), (self.ask_sizes, 48)):
            if type(vector) is not tuple:
                raise ValueError("Snapshot vectors must be immutable tuples")
            for value in vector:
                if checked(value, bits, False) == 0:
                    raise ValueError("Prices and quantities must be positive")
        if (self.bid_prices[0] >= self.ask_prices[0] or
                any(a <= b for a, b in zip(self.bid_prices, self.bid_prices[1:])) or
                any(a >= b for a, b in zip(self.ask_prices, self.ask_prices[1:]))):
            raise ValueError("Invalid, locked, or crossed book")

    @classmethod
    def from_snapshot(cls, row):
        return cls(row.timestamp_ns, row.session_id, row.symbol,
                   tuple(decimal_units(v, PRICE_UNITS) for v in row.bid_prices),
                   tuple(decimal_units(v, QUANTITY_UNITS) for v in row.bid_sizes),
                   tuple(decimal_units(v, PRICE_UNITS) for v in row.ask_prices),
                   tuple(decimal_units(v, QUANTITY_UNITS) for v in row.ask_sizes))


def read_fixed_snapshots(path, levels=5):
    result, seen = [], set()
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns(levels):
            raise ValueError("CSV header does not match configured depth")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(v is None or not v.strip() for v in row.values()):
                    raise ValueError("Missing or extra fields")
                vectors = [tuple(decimal_units(row[f"{name}_{i}"], units) for i in range(1, levels + 1))
                           for name, units in (("bid_price", PRICE_UNITS), ("bid_size", QUANTITY_UNITS),
                                               ("ask_price", PRICE_UNITS), ("ask_size", QUANTITY_UNITS))]
                book = FixedSnapshot(int(row["timestamp_ns"]), row["session_id"], row["symbol"], *vectors)
                if result:
                    if book.timestamp_ns <= result[-1].timestamp_ns or book.symbol != result[-1].symbol:
                        raise ValueError("Require increasing timestamps and one instrument")
                    if book.session_id != result[-1].session_id and book.session_id in seen:
                        raise ValueError("Sessions must form contiguous blocks")
                result.append(book)
                seen.add(book.session_id)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid fixed-point CSV row {line}: {exc}") from exc
    if not result:
        raise ValueError("Snapshot file is empty")
    return result


def constants(config, scaler, input_exponent):
    """Offline compilation of floating-point training metadata, never fitted anew."""
    if not 1 <= config.levels <= 32 or not 1 <= config.lookback <= 64:
        raise ValueError("Fixed frontend supports 1..32 levels and 1..64 history snapshots")
    if type(input_exponent) is not int or not -FEATURE_BITS <= input_exponent <= 31:
        raise ValueError("Input exponent must be between -24 and 31")
    names = feature_names(config)
    if len(scaler.mean) != len(names):
        raise ValueError("Preprocessing width differs from feature configuration")
    scaler.transform([0] * len(names))  # Validate existing fitted metadata.
    tick = decimal_units(config.tick_size, PRICE_UNITS)
    if checked(tick, 32, False) == 0:
        raise ValueError("Tick must occupy at least one price unit")
    with localcontext() as context:
        context.prec = 60
        def quantize(value, bits):
            return int((value * (1 << bits)).to_integral_value(rounding=ROUND_HALF_UP))
        centers = [checked(quantize(Decimal(str(v)), FEATURE_BITS), 64) for v in scaler.mean]
        inverses = [checked(quantize(1 / Decimal(str(v)), COEFFICIENT_BITS), 64, False) for v in scaler.scale]
        if any(v == 0 for v in inverses):
            raise ValueError("Normalization reciprocal underflows Q30")
        clip = None if scaler.clip is None else checked(quantize(Decimal(str(scaler.clip)), FEATURE_BITS), 64)
        if clip is not None and clip <= 0:
            raise ValueError("Preprocessing bound underflows Q24")
        lut = [quantize((1 + Decimal(i) / (1 << TABLE_BITS)).ln(), LOG_BITS)
               for i in range((1 << TABLE_BITS) + 1)]
        log_quantity_units = quantize(Decimal(QUANTITY_UNITS).ln(), LOG_BITS)
    return {"schema_version": 1, "format": FORMAT,
            "feature_config": {"levels": config.levels, "lookback": config.lookback,
                               "include_mid_changes": config.include_mid_changes,
                               "include_queue_features": config.include_queue_features},
            "feature_names": names, "price_units": PRICE_UNITS, "quantity_units": QUANTITY_UNITS,
            "feature_fraction_bits": FEATURE_BITS, "coefficient_fraction_bits": COEFFICIENT_BITS,
            "log_fraction_bits": LOG_BITS, "log_table_bits": TABLE_BITS,
            "tick_units": tick, "input_scale_exponent": input_exponent, "normalization_method": scaler.method,
            "centers_q24": centers, "inverse_scales_q30": inverses, "clip_q24": clip,
            "log_lut_q30": lut, "ln2_q30": lut[-1], "log_quantity_units_q30": log_quantity_units}


def compile_model(model):
    _, network, _, _ = validate_export(model)
    if network.get("task") != "lob_direction":
        raise ValueError("Fixed features require a LOB model")
    config = FeatureConfig(**network["feature_config"])
    if network["feature_names"] != feature_names(config):
        raise ValueError("Feature names differ from model configuration")
    scaler = Standardizer.from_dict(json.loads((model / "preprocessing.json").read_text()))
    document = constants(config, scaler, network["layers"][0]["input_scale_exponent"])
    document["source_model_report_sha256"] = hashlib.sha256((model / "report.json").read_bytes()).hexdigest()
    return document


def memory_words(document):
    return {"centers.mem": [f'{v & ((1 << 64) - 1):016x}' for v in document["centers_q24"]],
            "inverse_scales.mem": [f'{v:016x}' for v in document["inverse_scales_q30"]],
            "log_lut.mem": [f'{v:08x}' for v in document["log_lut_q30"]]}


def export_frontend(model, output):
    model, output = Path(model), Path(output)
    if output.resolve() == model.resolve() or output.resolve().is_relative_to(model.resolve()):
        raise ValueError("Export frontend separately from the frozen model")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Frontend output directory must be empty")
    document = compile_model(model)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frontend.json").write_text(json.dumps(document, indent=2) + "\n")
    for name, words in memory_words(document).items():
        (output / name).write_text("\n".join(words) + "\n")
    return document


class FixedPreprocessor:
    def __init__(self, document):
        # Public loading goes through load(), which recomputes and checks every
        # constant and ROM word against the hash-validated frozen model.
        if document.get("format") != FORMAT or document.get("schema_version") != 1:
            raise ValueError("Unsupported fixed-point frontend")
        self.document = document
        self.config = document["feature_config"]
        self.tick = document["tick_units"]
        self.exponent = document["input_scale_exponent"]
        self.lut = document["log_lut_q30"]

    @classmethod
    def load(cls, directory, model):
        directory, model = Path(directory), Path(model)
        actual = json.loads((directory / "frontend.json").read_text())
        expected = compile_model(model)
        if actual != expected:
            raise ValueError("Fixed frontend constants or model binding differ from frozen model")
        for name, words in memory_words(expected).items():
            if (directory / name).read_text().split() != words:
                raise ValueError(f"Fixed frontend ROM mismatch: {name}")
        return cls(expected)

    def log_integer_q30(self, value):
        checked(value, 49, False)
        if value == 0:
            raise ValueError("Logarithm requires a positive integer")
        exponent = value.bit_length() - 1
        mantissa = value << (32 - exponent) if exponent <= 32 else value >> (exponent - 32)
        fraction_bits = 32 - TABLE_BITS
        index = (mantissa - (1 << 32)) >> fraction_bits
        remainder = mantissa & ((1 << fraction_bits) - 1)
        interpolation = shift_round((self.lut[index + 1] - self.lut[index]) * remainder, fraction_bits)
        return exponent * self.document["ln2_q30"] + self.lut[index] + interpolation

    def log_quantity_q24(self, quantity):
        checked(quantity, 48, False)
        return checked(shift_round(self.log_integer_q30(quantity + QUANTITY_UNITS)
                                  - self.document["log_quantity_units_q30"], LOG_BITS - FEATURE_BITS), 64)

    def ratio_q24(self, numerator, denominator):
        return checked(divide_round(numerator << FEATURE_BITS, denominator), 64)

    def snapshot_features(self, row):
        if not isinstance(row, FixedSnapshot) or len(row.bid_prices) != self.config["levels"]:
            raise ValueError("Require a fixed-point snapshot at the configured depth")
        mid2 = row.bid_prices[0] + row.ask_prices[0]
        values = []
        for bid, ask, bid_size, ask_size in zip(row.bid_prices, row.ask_prices, row.bid_sizes, row.ask_sizes):
            values.extend((self.ratio_q24(mid2 - 2 * bid, 2 * self.tick),
                           self.ratio_q24(2 * ask - mid2, 2 * self.tick),
                           self.log_quantity_q24(bid_size), self.log_quantity_q24(ask_size)))
        bid_depth, ask_depth = sum(row.bid_sizes), sum(row.ask_sizes)
        return values + [self.ratio_q24(row.ask_prices[0] - row.bid_prices[0], self.tick),
                         self.ratio_q24(bid_depth - ask_depth, bid_depth + ask_depth)]

    def feature_window(self, rows, decision):
        start = decision - self.config["lookback"] + 1
        if start < 0 or decision >= len(rows):
            raise ValueError("Insufficient feature history")
        window = rows[start:decision + 1]
        if any(not isinstance(row, FixedSnapshot) for row in window):
            raise ValueError("Runtime frontend accepts integer snapshots only")
        if any((row.session_id, row.symbol) != (window[0].session_id, window[0].symbol) for row in window):
            raise ValueError("Feature window crosses session or instrument")
        if any(a.timestamp_ns >= b.timestamp_ns for a, b in zip(window, window[1:])):
            raise ValueError("Feature timestamps must strictly increase")
        values = [v for row in window for v in self.snapshot_features(row)]
        if self.config["include_mid_changes"]:
            values += [self.ratio_q24(b.bid_prices[0] + b.ask_prices[0] - a.bid_prices[0] - a.ask_prices[0], 2 * self.tick)
                       for a, b in zip(window, window[1:])]
        if self.config["include_queue_features"]:
            for row in window:
                delta, total = row.bid_sizes[0] - row.ask_sizes[0], row.bid_sizes[0] + row.ask_sizes[0]
                values += [self.ratio_q24(delta, total),
                           self.ratio_q24((row.ask_prices[0] - row.bid_prices[0]) * delta, 2 * self.tick * total)]
            for before, after in zip(window, window[1:]):
                flow = ((after.bid_prices[0] >= before.bid_prices[0]) * after.bid_sizes[0]
                        - (after.bid_prices[0] <= before.bid_prices[0]) * before.bid_sizes[0]
                        - (after.ask_prices[0] <= before.ask_prices[0]) * after.ask_sizes[0]
                        + (after.ask_prices[0] >= before.ask_prices[0]) * before.ask_sizes[0])
                total = before.bid_sizes[0] + before.ask_sizes[0] + after.bid_sizes[0] + after.ask_sizes[0]
                values.append(self.ratio_q24(2 * flow, total))
        return values

    def normalize(self, features):
        if len(features) != len(self.document["centers_q24"]):
            raise ValueError("Feature count differs from normalization coefficients")
        result = []
        for value, center, reciprocal in zip(features, self.document["centers_q24"], self.document["inverse_scales_q30"]):
            checked(value, 64)
            product = checked((value - center) * reciprocal, 128)
            normalized = checked(shift_round(product, COEFFICIENT_BITS), 64)
            clip = self.document["clip_q24"]
            if clip is not None:
                normalized = max(-clip, min(clip, normalized))
            result.append(normalized)
        return result

    def quantize(self, normalized):
        if len(normalized) != len(self.document["centers_q24"]):
            raise ValueError("Normalized feature count differs")
        return [max(-128, min(127, shift_round(checked(value, 64), FEATURE_BITS + self.exponent))) for value in normalized]

    def inputs(self, rows, decision):
        return self.quantize(self.normalize(self.feature_window(rows, decision)))
