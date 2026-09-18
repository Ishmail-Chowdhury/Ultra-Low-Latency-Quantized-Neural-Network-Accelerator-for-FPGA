"""Bit-exact signed INT8 dense layer; uses only the Python standard library."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence


def signed_range(bits: int) -> tuple[int, int]:
    if type(bits) is not int or bits < 2:
        raise ValueError("signed width must be an integer >= 2")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def checked_int(value: int, bits: int, name: str) -> int:
    low, high = signed_range(bits)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be a signed {bits}-bit integer: {value!r}")
    return value


def requantize(acc: int, shift: int, relu: bool = False, acc_bits: int = 32) -> int:
    """Round to nearest, ties away from zero; apply optional ReLU and INT8 clip."""
    checked_int(acc, acc_bits, "accumulator")
    if type(shift) is not int or not 0 <= shift < acc_bits:
        raise ValueError("shift must be an integer in [0, acc_bits)")
    if type(relu) is not bool:
        raise ValueError("relu must be a boolean")
    magnitude = abs(acc)
    if shift:
        magnitude = (magnitude + (1 << (shift - 1))) >> shift
    rounded = -magnitude if acc < 0 else magnitude
    return max(0 if relu else -128, min(127, rounded))


@dataclass(frozen=True)
class DenseSpec:
    weights: tuple[tuple[int, ...], ...]
    biases: tuple[int, ...]
    shift: int = 7
    relu: bool = False
    acc_bits: int = 32

    def __post_init__(self) -> None:
        if type(self.acc_bits) is not int or self.acc_bits < 16:
            raise ValueError("acc_bits must be an integer >= 16")
        requantize(0, self.shift, self.relu, self.acc_bits)
        if not self.weights or not self.weights[0]:
            raise ValueError("dense dimensions must be positive")
        if len(self.biases) != len(self.weights):
            raise ValueError("one bias is required per output")
        max_acc = signed_range(self.acc_bits)[1]
        for row, bias in zip(self.weights, self.biases):
            if len(row) != self.inputs:
                raise ValueError("weight rows must have equal lengths")
            checked_int(bias, self.acc_bits, "bias")
            for weight in row:
                checked_int(weight, 8, "weight")
            if abs(bias) + 128 * sum(abs(w) for w in row) > max_acc:
                raise ValueError("weights and bias can overflow the accumulator")

    @property
    def inputs(self) -> int:
        return len(self.weights[0])

    @property
    def outputs(self) -> int:
        return len(self.weights)

    def evaluate(self, inputs: Sequence[int]) -> tuple[list[int], list[int]]:
        if len(inputs) != self.inputs:
            raise ValueError(f"expected {self.inputs} inputs, got {len(inputs)}")
        for value in inputs:
            checked_int(value, 8, "input")
        accumulators = []
        for row, bias in zip(self.weights, self.biases):
            acc = bias
            for value, weight in zip(inputs, row):
                acc = checked_int(acc + value * weight, self.acc_bits, "partial sum")
            accumulators.append(acc)
        return accumulators, [
            requantize(a, self.shift, self.relu, self.acc_bits) for a in accumulators
        ]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "data_bits": 8,
            "acc_bits": self.acc_bits,
            "shift": self.shift,
            "relu": self.relu,
            "zero_point": 0,
            "rounding": "nearest_ties_away_from_zero",
            "weights": [list(row) for row in self.weights],
            "biases": list(self.biases),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DenseSpec:
        conventions = {
            "schema_version": 1, "data_bits": 8, "zero_point": 0,
            "rounding": "nearest_ties_away_from_zero",
        }
        if any(data.get(k) != v for k, v in conventions.items()):
            raise ValueError("unsupported model arithmetic or schema")
        spec = cls(tuple(tuple(row) for row in data["weights"]),
                   tuple(data["biases"]), data["shift"], data["relu"], data["acc_bits"])
        if data.get("inputs") != spec.inputs or data.get("outputs") != spec.outputs:
            raise ValueError("model dimensions do not match weights")
        return spec


def write_mem(path: Path, values: Sequence[int], bits: int) -> None:
    for value in values:
        checked_int(value, bits, "memory value")
    path.write_text("".join(f"{v & ((1 << bits) - 1):0{(bits + 3) // 4}x}\n"
                            for v in values), encoding="ascii")


def export_model(spec: DenseSpec, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    write_mem(output / "weights.mem", [w for row in spec.weights for w in row], 8)
    write_mem(output / "biases.mem", spec.biases, spec.acc_bits)


def fixture(seed: int = 2026) -> tuple[DenseSpec, list[list[int]]]:
    rng = random.Random(seed)
    spec = DenseSpec(tuple(tuple(rng.randint(-128, 127) for _ in range(8))
                           for _ in range(4)), (0, 127, -128, 1024))
    vectors = [[0] * 8, [-128] * 8, [127] * 8, [-128, 127] * 4]
    vectors += [[rng.randint(-128, 127) for _ in range(8)] for _ in range(32)]
    return spec, vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/fixture"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    spec, inputs = fixture(args.seed)
    export_model(spec, args.output)
    vectors = []
    for values in inputs:
        accumulators, outputs = spec.evaluate(values)
        vectors.append({"inputs": values, "accumulators": accumulators, "outputs": outputs})
    (args.output / "vectors.json").write_text(
        json.dumps({"schema_version": 1, "seed": args.seed, "vectors": vectors}, indent=2) + "\n")
    print(f"Exported {spec.inputs}x{spec.outputs} model and {len(vectors)} vectors to {args.output}")


if __name__ == "__main__":
    main()
