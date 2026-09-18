import random
from pathlib import Path
import tempfile
import unittest

from models.quantization.dense_reference import DenseSpec, export_model, fixture
from tb.dense_runner import DENSE_SOURCES, verify_dense
from tb.sim import simulate


class DenseTests(unittest.TestCase):
    def test_malformed_frame_assertion(self):
        spec, _ = fixture()
        for early in [False, True]:
            bench = f"""
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, s_tvalid = 0, s_tlast = 0, m_tready = 1;
    reg signed [7:0] s_tdata = 0;
    wire s_tready, m_tvalid, m_tlast;
    wire signed [7:0] m_tdata;
    integer i;
    dense_layer dut (.*);
    initial begin
        repeat (2) @(negedge clk);
        rst = 0;
        for (i = 0; i < 8; i = i + 1) begin
            s_tvalid = 1; s_tlast = {int(early)} && i == 0;
            @(negedge clk);
        end
        $finish;
    end
endmodule
"""
            with self.subTest(early=early), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                export_model(spec, path)
                simulate(path, bench, DENSE_SOURCES, expect_failure=True,
                         failure_message="Input tlast does not match vector length")

    def test_fixture_and_cycle_count(self):
        spec, vectors = fixture()
        result = verify_dense(spec, vectors, stalls=False)
        self.assertEqual(result["outputs"], 144)
        self.assertEqual(result["last_input_to_last_output_cycles"], 44)
        self.assertEqual(result["first_input_to_last_output_cycles"], 51)

    def test_backpressure(self):
        spec, vectors = fixture()
        for seed in [1, 42, 2026]:
            with self.subTest(seed=seed):
                verify_dense(spec, vectors, seed=seed)

    def test_reset_discards_inflight_vector(self):
        spec, vectors = fixture()
        for phase in [1, 2, 3]:
            with self.subTest(phase=phase):
                verify_dense(spec, vectors[:5], reset_phase=phase)

    def test_dimensions_relu_and_shifts(self):
        rng = random.Random(99)
        for inputs, outputs, shift, relu, bits in [(1, 1, 0, False, 16),
                                                  (3, 5, 1, True, 32),
                                                  (7, 2, 31, False, 32),
                                                  (8, 4, 7, True, 32)]:
            spec = DenseSpec(tuple(tuple(rng.randint(-128, 127) for _ in range(inputs))
                                   for _ in range(outputs)), (0,) * outputs, shift, relu, bits)
            vectors = [[-128] * inputs, [127] * inputs, [0] * inputs]
            vectors += [[rng.randint(-128, 127) for _ in range(inputs)] for _ in range(20)]
            for stalls in [False, True]:
                with self.subTest(inputs=inputs, outputs=outputs, stalls=stalls):
                    verify_dense(spec, vectors, stalls=stalls)
