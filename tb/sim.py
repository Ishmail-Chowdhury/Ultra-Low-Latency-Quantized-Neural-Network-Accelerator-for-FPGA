"""Compile and execute temporary SystemVerilog testbenches with Icarus."""

import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def tool(name: str) -> str:
    configured = os.environ.get(name.upper())
    local = ROOT / "build" / "simulator" / "bin" / name
    found = configured or shutil.which(name) or (str(local) if local.is_file() else None)
    if not found:
        raise RuntimeError(f"{name} is required for RTL tests; see README.md")
    return found


def simulate(directory: Path, bench: str, sources: list[str], *, expect_failure=False) -> str:
    (directory / "tb.sv").write_text(bench)
    compiled = subprocess.run(
        [tool("iverilog"), "-g2012", "-Wall", "-s", "tb", "-o", "sim.vvp",
         *(str(ROOT / source) for source in sources), "tb.sv"],
        cwd=directory, text=True, capture_output=True, timeout=60,
    )
    if compiled.returncode:
        raise AssertionError(f"RTL compilation failed:\n{compiled.stdout}{compiled.stderr}")
    result = subprocess.run([tool("vvp"), "sim.vvp"], cwd=directory, text=True,
                            capture_output=True, timeout=60)
    output = result.stdout + result.stderr
    if expect_failure:
        if result.returncode == 0 or "MAC accumulator overflow" not in output:
            raise AssertionError(f"Expected overflow assertion did not fire:\n{output}")
    elif result.returncode or "PASS" not in output:
        raise AssertionError(f"RTL simulation failed:\n{output}")
    return output
