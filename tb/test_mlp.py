import random
import unittest

from models.quantization.dense_reference import DenseSpec
from tb.dense_runner import verify_mlp


class MlpTests(unittest.TestCase):
    def test_parallel_layers_and_partial_groups(self):
        rng = random.Random(3070)
        first = DenseSpec(tuple(tuple(rng.randint(-128, 127) for _ in range(3)) for _ in range(5)),
                          (1, -200, 300, -400, 500), 6, True)
        second = DenseSpec(tuple(tuple(rng.randint(-128, 127) for _ in range(5)) for _ in range(7)),
                           (700, -600, 500, -400, 300, -200, 100), 7, False)
        vectors = [[rng.randint(-128, 127) for _ in range(3)] for _ in range(20)]
        # Fully parallel hidden neurons feeding a serial, slower output layer
        # also exercise internal stalls over repeated vectors.
        for hidden_lanes, output_lanes in [(2, 3), (5, 1), (1, 7), (5, 7)]:
            for stalls in [False, True]:
                with self.subTest(hidden_lanes=hidden_lanes, output_lanes=output_lanes, stalls=stalls):
                    verify_mlp(first, second, vectors, hidden_lanes=hidden_lanes,
                               output_lanes=output_lanes, stalls=stalls)
            for phase in [1, 2, 3, 4]:
                with self.subTest(hidden_lanes=hidden_lanes, output_lanes=output_lanes, phase=phase):
                    verify_mlp(first, second, vectors[:4], hidden_lanes=hidden_lanes,
                               output_lanes=output_lanes, reset_phase=phase)

    def test_parallel_single_vector_latency(self):
        first = DenseSpec(((1, 2), (3, 4), (5, 6)), (0, 0, 0), 0, True)
        second = DenseSpec(((1, -1, 1), (-1, 1, -1)), (0, 0), 0, False)
        result = verify_mlp(first, second, [[1, 2]], stalls=False, hidden_lanes=2, output_lanes=2)
        self.assertEqual(result["last_input_to_last_output_cycles"], 2 * (2 + 2) + 3 + (3 + 2) + 2)

    def test_chained_layers_and_internal_backpressure(self):
        rng = random.Random(300)
        # A slow second layer forces stalls between the two RTL modules.
        first = DenseSpec(((20, -30, 40), (-10, 80, -60)), (127, -128), 5, True)
        second = DenseSpec(tuple((rng.randint(-128, 127), rng.randint(-128, 127))
                                 for _ in range(7)), (0,) * 7, 7, False)
        vectors = [[rng.randint(-128, 127) for _ in range(3)] for _ in range(30)]
        for stalls in [False, True]:
            with self.subTest(stalls=stalls):
                result = verify_mlp(first, second, vectors, stalls=stalls)
                self.assertEqual(result["outputs"], 210)
        for phase in [1, 2, 3]:
            with self.subTest(reset_phase=phase):
                verify_mlp(first, second, vectors[:4], reset_phase=phase)

    def test_single_vector_latency(self):
        first = DenseSpec(((1, 2), (3, 4)), (0, 0), 0, True)
        second = DenseSpec(((1, -1),), (0,), 0, False)
        result = verify_mlp(first, second, [[1, 2]], stalls=False)
        self.assertEqual(result["last_input_to_last_output_cycles"], 2 * (2 + 3) + 1 * (2 + 3))
