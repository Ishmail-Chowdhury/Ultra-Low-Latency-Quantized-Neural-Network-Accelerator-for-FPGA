import random
import unittest

from models.quantization.dense_reference import DenseSpec, fixture
from tb.dense_runner import verify_dense


class DenseTests(unittest.TestCase):
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
