import random
import unittest

from models.quantization.dense_reference import DenseSpec
from tb.dense_runner import verify_mlp


class MlpTests(unittest.TestCase):
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
