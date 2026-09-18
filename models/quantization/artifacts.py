"""Validate saved model files, memory contents, and arithmetic metadata together."""

import hashlib
import json
from pathlib import Path
import re

from models.quantization.dense_reference import DenseSpec


def read_signed_memory(path: Path, bits: int) -> list[int]:
    values = []
    for token in path.read_text().split():
        if not re.fullmatch(r"[0-9a-fA-F]+", token) or len(token) != (bits + 3) // 4:
            raise ValueError(f"Invalid memory word in {path.name}")
        value = int(token, 16)
        if value >= 1 << bits:
            raise ValueError(f"Memory word exceeds {bits} bits")
        values.append(value - (1 << bits) if value & (1 << (bits - 1)) else value)
    return values


def validate_export(directory: Path) -> tuple[dict, dict, DenseSpec, DenseSpec]:
    report = json.loads((directory / "report.json").read_text())
    network = json.loads((directory / "network.json").read_text())
    if report.get("schema_version") != 1 or network.get("schema_version") != 1:
        raise ValueError("Unsupported export schema")
    required = {"network.json", "float_parameters.json", "split.json", "vectors.json"}
    required.update(f"layer{i}/{name}" for i in (0, 1)
                    for name in ("model.json", "weights.mem", "biases.mem"))
    if network.get("task") == "lob_direction":
        required.add("preprocessing.json")
        if network.get("dataset_kind") != report.get("dataset_kind"):
            raise ValueError("Inconsistent dataset provenance")
    hashes = report.get("sha256")
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError("Incomplete export checksum manifest")
    for name, expected in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Artifact path leaves export directory")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Export checksum mismatch: {name}")
    layers = [DenseSpec.from_dict(json.loads((directory / f"layer{i}/model.json").read_text()))
              for i in (0, 1)]
    first, second = layers
    architecture = [first.inputs, first.outputs, second.outputs]
    if network.get("architecture") != architecture or report.get("architecture") != architecture:
        raise ValueError("Export architecture does not match layer dimensions")
    if first.outputs != second.inputs or first.acc_bits != second.acc_bits or not first.relu or second.relu:
        raise ValueError("Invalid two-layer MLP arithmetic")
    if len(network.get("classes", [])) != second.outputs:
        raise ValueError("Class count does not match output layer")
    scales = network.get("layers", [])
    if len(scales) != 2:
        raise ValueError("Two layer scale records are required")
    for index, (spec, scale) in enumerate(zip(layers, scales)):
        keys = ("input_scale_exponent", "weight_scale_exponent", "bias_scale_exponent", "output_scale_exponent")
        if any(type(scale.get(key)) is not int for key in keys):
            raise ValueError("Scale exponents must be integers")
        if (scale.get("directory") != f"layer{index}" or
                scale["bias_scale_exponent"] != scale["input_scale_exponent"] + scale["weight_scale_exponent"] or
                scale["output_scale_exponent"] - scale["bias_scale_exponent"] != spec.shift):
            raise ValueError("Scale metadata does not match layer arithmetic")
        if index and scale["input_scale_exponent"] != scales[index - 1]["output_scale_exponent"]:
            raise ValueError("Adjacent layer scales do not match")
        path = directory / f"layer{index}"
        if read_signed_memory(path / "weights.mem", 8) != [w for row in spec.weights for w in row]:
            raise ValueError("Weight memory does not match model JSON")
        if read_signed_memory(path / "biases.mem", spec.acc_bits) != list(spec.biases):
            raise ValueError("Bias memory does not match model JSON")
    return report, network, first, second
