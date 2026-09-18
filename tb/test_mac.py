from pathlib import Path
import random
import tempfile
import unittest

from models.quantization.dense_reference import requantize, write_mem
from tb.sim import simulate


MAC_HEADER = """
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, load = 0, enable = 0;
    reg signed [7:0] x = 0, weight = 0;
    reg signed [31:0] bias = 0;
    wire signed [31:0] acc;
    mac dut (.*);
    task step(input bit r, l, e, input integer a, w, b, expected);
        @(negedge clk);
        rst = r; load = l; enable = e; x = a; weight = w; bias = b;
        @(posedge clk); #1;
        if (acc !== expected) $fatal(1, "MAC got %0d expected %0d", acc, expected);
    endtask
"""


class MacTests(unittest.TestCase):
    def test_all_int8_products_and_accumulation(self):
        rng = random.Random(2026)
        steps = []
        for _ in range(32):
            acc = rng.randint(-100000, 100000)
            steps.append(f"step(0, 1, 0, 0, 0, {acc}, {acc});")
            for _ in range(64):
                x, w = rng.randint(-128, 127), rng.randint(-128, 127)
                enable = rng.randrange(2)
                acc += x * w if enable else 0
                steps.append(f"step(0, 0, {enable}, {x}, {w}, 0, {acc});")
        bench = MAC_HEADER + """
    integer a, w;
    initial begin
        step(1, 1, 1, 127, 127, 999, 0);
        for (a = -128; a <= 127; a = a + 1)
            for (w = -128; w <= 127; w = w + 1) begin
                step(0, 1, 1, a, w, 0, 0);
                step(0, 0, 1, a, w, 0, a*w);
            end
        step(0, 1, 0, 0, 0, 2147483647, 2147483647);
        step(0, 0, 1, 1, -1, 0, 2147483646);
        step(0, 1, 0, 0, 0, -2147483648, -2147483648);
        step(0, 0, 1, 1, 1, 0, -2147483647);
        """ + "\n".join(steps) + """
        step(1, 0, 1, -128, -128, 0, 0);
        step(0, 0, 0, -128, -128, 0, 0);
        $display("PASS: all 65536 INT8 products, accumulation, hold, load, reset");
        $finish;
    end
endmodule
"""
        with tempfile.TemporaryDirectory() as directory:
            simulate(Path(directory), bench, ["rtl/compute/mac.sv"])

    def test_overflow_assertion(self):
        for bias, weight in [(2147483647, 1), (-2147483648, -1)]:
            bench = MAC_HEADER + f"""
    initial begin
        step(1, 0, 0, 0, 0, 0, 0);
        step(0, 1, 0, 0, 0, {bias}, {bias});
        @(negedge clk); load = 0; enable = 1; x = 1; weight = {weight};
        @(posedge clk); #1; $finish;
    end
endmodule
"""
            with self.subTest(bias=bias), tempfile.TemporaryDirectory() as directory:
                simulate(Path(directory), bench, ["rtl/compute/mac.sv"], expect_failure=True)


class RequantizationTests(unittest.TestCase):
    def test_boundaries_and_random_values(self):
        rng = random.Random(77)
        for bits in [16, 32]:
            low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            for shift in [0, 1, 7, bits - 1]:
                values = [low, low + 1, high, high - 1, 0, 1, -1, 127, -128, 128, -129]
                if shift:
                    for multiplier in [0, 1, 126, 127, 128]:
                        boundary = (multiplier << shift) + (1 << (shift - 1))
                        values += [sign * (boundary + offset)
                                   for sign in [-1, 1] for offset in [-1, 0, 1]
                                   if low <= sign * (boundary + offset) <= high]
                values += [rng.randint(low, high) for _ in range(1000)]
                for relu in [False, True]:
                    expected = [requantize(x, shift, relu, bits) for x in values]
                    bench = f"""
`timescale 1ns/1ps
module tb;
    reg signed [{bits - 1}:0] acc;
    wire signed [7:0] result;
    reg signed [{bits - 1}:0] inputs [0:{len(values)-1}];
    reg signed [7:0] outputs [0:{len(values)-1}];
    integer i;
    requantize #(.ACC_WIDTH({bits}), .SHIFT({shift}), .RELU({int(relu)})) dut (.*);
    initial begin
        $readmemh("inputs.mem", inputs);
        $readmemh("outputs.mem", outputs);
        for (i = 0; i < {len(values)}; i = i + 1) begin
            acc = inputs[i]; #1;
            if (result !== outputs[i])
                $fatal(1, "requantize input %0d got %0d expected %0d", acc, result, outputs[i]);
        end
        $display("PASS: requantization"); $finish;
    end
endmodule
"""
                    with self.subTest(bits=bits, shift=shift, relu=relu), tempfile.TemporaryDirectory() as directory:
                        path = Path(directory)
                        write_mem(path / "inputs.mem", values, bits)
                        write_mem(path / "outputs.mem", expected, 8)
                        simulate(path, bench, ["rtl/compute/requantize.sv"])
