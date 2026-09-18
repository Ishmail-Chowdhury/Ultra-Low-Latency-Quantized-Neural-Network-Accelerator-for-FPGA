"""Reusable streaming scoreboard for synthetic and trained model layers."""

from pathlib import Path
import re
import shutil
import tempfile

from models.quantization.dense_reference import DenseSpec, export_model, write_mem
from tb.sim import simulate

DENSE_SOURCES = ["rtl/compute/mac.sv", "rtl/compute/requantize.sv", "rtl/compute/dense_layer.sv"]


def verify_dense(spec: DenseSpec, vectors: list[list[int]], *, stalls: bool = True,
                 seed: int = 2026, reset_phase: int = 0, memory_directory: Path | None = None) -> dict:
    return verify_stream(spec, vectors, stalls=stalls, seed=seed, reset_phase=reset_phase,
                         memory_directory=memory_directory)


def verify_mlp(first: DenseSpec, second: DenseSpec, vectors: list[list[int]], *,
               stalls: bool = True, seed: int = 2026, reset_phase: int = 0,
               memory_directory: Path | None = None) -> dict:
    if first.outputs != second.inputs or first.acc_bits != second.acc_bits:
        raise ValueError("MLP dimensions and accumulator widths must agree")
    if not first.relu or second.relu:
        raise ValueError("MLP requires hidden ReLU and unrectified output logits")
    return verify_stream(first, vectors, second=second, stalls=stalls, seed=seed, reset_phase=reset_phase,
                         memory_directory=memory_directory)


def verify_stream(spec: DenseSpec, vectors: list[list[int]], *, second: DenseSpec | None = None,
                  stalls: bool = True, seed: int = 2026, reset_phase: int = 0,
                  memory_directory: Path | None = None) -> dict:
    """reset_phase: 0=none, 1=partial input, 2=compute, 3=stalled output."""
    if not vectors:
        raise ValueError("at least one vector is required")
    if reset_phase not in (0, 1, 2, 3):
        raise ValueError("invalid reset phase")
    layer_outputs = [spec.evaluate(x)[1] for x in vectors]
    if second is not None:
        layer_outputs = [second.evaluate(x)[1] for x in layer_outputs]
    expected = [value for values in layer_outputs for value in values]
    outputs = second.outputs if second else spec.outputs
    sources = list(DENSE_SOURCES)
    cycles_per_vector = spec.outputs * (spec.inputs + 3)
    if second is None:
        instance = f"""dense_layer #(.INPUTS({spec.inputs}), .OUTPUTS({spec.outputs}),
            .ACC_WIDTH({spec.acc_bits}), .SHIFT({spec.shift}), .RELU({int(spec.relu)})) dut (.*);"""
    else:
        sources.append("rtl/top/mlp.sv")
        cycles_per_vector += second.outputs * (second.inputs + 3)
        instance = f"""mlp #(.INPUTS({spec.inputs}), .HIDDEN({spec.outputs}), .OUTPUTS({outputs}),
            .ACC_WIDTH({spec.acc_bits}), .HIDDEN_SHIFT({spec.shift}), .OUTPUT_SHIFT({second.shift})) dut (.*);"""
    flat_inputs = [value for x in vectors for value in x]
    bench = f"""
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1;
    reg signed [7:0] s_tdata = 0;
    reg s_tvalid = 0, s_tlast = 0, m_tready = 0;
    wire s_tready, m_tvalid, m_tlast;
    wire signed [7:0] m_tdata;
    reg signed [7:0] input_mem [0:{len(flat_inputs)-1}];
    reg signed [7:0] expected_mem [0:{len(expected)-1}];
    integer sent = 0, received = 0, cycles = 0;
    integer first_input_cycles [0:{len(vectors)-1}];
    integer final_input_cycles [0:{len(vectors)-1}];
    integer frame_latency = 0, first_latency = 0;
    integer min_latency = 2147483647, max_latency = 0;
    integer stall_cycles = 0, warmup_sent = 0;
    integer drain_cycles = 0;
    reg running = 0, held = 0, input_accepted = 0;
    reg signed [7:0] held_data;
    reg held_last;
    reg [31:0] rng = 32'd{seed & 0xffffffff};

    {instance}

    initial begin
        $readmemh("inputs.mem", input_mem);
        $readmemh("expected.mem", expected_mem);
        repeat (3) @(negedge clk);
        if ({reset_phase} != 0) begin
            rst = 0;
            // Send a partial or full disposable frame; reset before any output transfers.
            while (warmup_sent < (({reset_phase} == 1) ? 1 : {spec.inputs})) begin
                s_tvalid = 1;
                s_tdata = input_mem[warmup_sent];
                s_tlast = (warmup_sent == {spec.inputs - 1});
                @(posedge clk);
                if (s_tready) warmup_sent = warmup_sent + 1;
                @(negedge clk);
            end
            s_tvalid = 0;
            if ({reset_phase} == 2) repeat (3) @(negedge clk);
            if ({reset_phase} == 3) begin
                while (!m_tvalid) @(negedge clk);
                repeat (12) @(negedge clk);
            end
            rst = 1;
            repeat (2) @(negedge clk);
        end
        rst = 0; running = 1;
    end

    always @(negedge clk) begin
        if (running) begin
            rng = rng * 32'd1664525 + 32'd1013904223;
            // Include a long stall, plus seeded intermittent downstream stalls.
            m_tready = !{int(stalls)} || (cycles > 80 && rng[17:16] != 0);
            // A source must hold its valid/data/last until the handshake.
            if (!s_tvalid || input_accepted) begin
                s_tvalid = sent < {len(flat_inputs)} && (!{int(stalls)} || rng[25:24] != 0);
                if (sent < {len(flat_inputs)}) begin
                    s_tdata = input_mem[sent];
                    s_tlast = (sent % {spec.inputs}) == {spec.inputs - 1};
                end
            end
        end
    end

    always @(posedge clk) begin
        cycles = cycles + 1;
        if (cycles > {max(10000, len(vectors) * (spec.inputs + cycles_per_vector) * 8)})
            $fatal(1, "Streaming test timed out");
        if (rst) begin
            held = 0;
            input_accepted = 0;
        end else if (running) begin
            input_accepted = s_tvalid && s_tready;
            if (held && (!m_tvalid || m_tdata !== held_data || m_tlast !== held_last))
                $fatal(1, "Scoreboard: unstable stalled output");
            held = m_tvalid && !m_tready;
            held_data = m_tdata; held_last = m_tlast;
            if (m_tvalid && !m_tready) stall_cycles = stall_cycles + 1;
            if (s_tvalid && s_tready) begin
                if ((sent % {spec.inputs}) == 0) first_input_cycles[sent / {spec.inputs}] = cycles;
                if (s_tlast) final_input_cycles[sent / {spec.inputs}] = cycles;
                sent = sent + 1;
            end
            if (m_tvalid && m_tready) begin
                if (received >= {len(expected)}) $fatal(1, "Extra output");
                if (m_tdata !== expected_mem[received])
                    $fatal(1, "Output %0d got %0d expected %0d", received, m_tdata, expected_mem[received]);
                if (m_tlast !== ((received % {outputs}) == {outputs - 1}))
                    $fatal(1, "Incorrect output tlast at %0d", received);
                if (m_tlast) begin
                    frame_latency = cycles - final_input_cycles[received / {outputs}];
                    first_latency = cycles - first_input_cycles[received / {outputs}];
                    if (frame_latency < min_latency) min_latency = frame_latency;
                    if (frame_latency > max_latency) max_latency = frame_latency;
                    if (!{int(stalls)} && {int(second is None)} && frame_latency != {cycles_per_vector})
                        $fatal(1, "Unexpected compute cycle count: %0d", frame_latency);
                end
                received = received + 1;
            end
            if (received == {len(expected)}) begin
                drain_cycles = drain_cycles + 1;
                if (drain_cycles == 20) begin
                    if (sent != {len(flat_inputs)}) $fatal(1, "Not all inputs transferred");
                    if ({int(stalls)} && stall_cycles == 0) $fatal(1, "Stall test did not stall");
                    $display("PASS: outputs=%0d last_input_to_last_output=%0d first_input_to_last_output=%0d stalls=%0d min_latency=%0d max_latency=%0d", received, frame_latency, first_latency, stall_cycles, min_latency, max_latency);
                    $finish;
                end
            end
        end
    end
endmodule
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        if second is None:
            export_model(spec, path)
        else:
            export_model(spec, path / "layer0")
            export_model(second, path / "layer1")
        if memory_directory is not None:
            # Exercise the user's saved ROM files, not newly generated substitutes.
            for subdir in ([""] if second is None else ["layer0", "layer1"]):
                for name in ("weights.mem", "biases.mem"):
                    shutil.copyfile(memory_directory / subdir / name, path / subdir / name)
        write_mem(path / "inputs.mem", flat_inputs, 8)
        write_mem(path / "expected.mem", expected, 8)
        output = simulate(path, bench, sources)
    match = re.search(r"outputs=(\d+) last_input_to_last_output=(\d+) first_input_to_last_output=(\d+) stalls=(\d+) min_latency=(\d+) max_latency=(\d+)", output)
    if not match:
        raise AssertionError(output)
    return dict(zip(["outputs", "last_input_to_last_output_cycles", "first_input_to_last_output_cycles", "stall_cycles",
                     "min_last_input_to_last_output_cycles", "max_last_input_to_last_output_cycles"],
                    map(int, match.groups())))
