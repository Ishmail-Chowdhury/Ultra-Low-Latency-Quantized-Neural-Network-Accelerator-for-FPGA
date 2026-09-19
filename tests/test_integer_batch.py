import importlib.util
import random
import unittest

from models.quantization.calibration import quantize_value
from models.quantization.dense_reference import DenseSpec


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional training checks require NumPy")
class IntegerBatchTests(unittest.TestCase):
    def test_signed_rounding_saturation_and_extreme_accumulators(self):
        from models.training.integer_batch import dense_batch
        rng = random.Random(2030)
        for shift in (0, 1, 7, 31):
            for relu in (False, True):
                # DenseSpec deliberately uses a conservative symmetric bound.
                spec = DenseSpec(((1, -1), (0, 0), (0, 0)), (0, -(2**31) + 1, 2**31 - 1), shift, relu)
                vectors = [[-128, 127], [127, -128], [0, 0]] + [[rng.randint(-128, 127) for _ in range(2)] for _ in range(30)]
                acc, output = dense_batch(spec, vectors)
                for i, x in enumerate(vectors):
                    self.assertEqual((acc[i].tolist(), output[i].tolist()), spec.evaluate(x))

    def test_input_quantization_ties_and_clipping(self):
        from models.training.integer_batch import quantize_inputs
        values = [[-129., -128.5, -2.5, -.5, 0., .5, 2.5, 127.5, 999.]]
        for exponent in (-2, 0, 2):
            self.assertEqual(quantize_inputs(values, exponent).tolist(),
                             [[quantize_value(v, exponent) for v in values[0]]])
