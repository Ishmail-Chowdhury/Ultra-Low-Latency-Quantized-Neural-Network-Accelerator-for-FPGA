import unittest

from models.quantization.calibration import quantize_dense, quantize_value, round_away, scale_exponent


class CalibrationTests(unittest.TestCase):
    def test_float_ties(self):
        for value, expected in [(0.5, 1), (-0.5, -1), (1.5, 2), (-1.5, -2), (0.49, 0)]:
            self.assertEqual(round_away(value), expected)
        self.assertEqual(quantize_value(1000, 0), 127)
        self.assertEqual(quantize_value(-1000, 0), -128)
        self.assertEqual(quantize_value(0.25, -3), 2)

    def test_scale_covers_maximum(self):
        for maximum in [0.01, 1, 3, 127, 128, 999]:
            exponent = scale_exponent(maximum)
            self.assertGreaterEqual(127 * 2.0 ** exponent, maximum)
            self.assertLess(127 * 2.0 ** (exponent - 1), maximum)
        self.assertEqual(scale_exponent(0), 0)

    def test_bias_scale_and_shift(self):
        model, scales = quantize_dense([[0.5, -0.25]], [0.125], -3, -2, True)
        self.assertEqual(scales["weight_scale_exponent"], -7)
        self.assertEqual(model.weights, ((64, -32),))
        self.assertEqual(model.biases, (128,))
        self.assertEqual(model.shift, 8)
        # Real dot product: 1 * .5 + .5 * -.25 + .125 = .5; output units .25.
        self.assertEqual(model.evaluate([8, 4]), ([512], [2]))

    def test_invalid_values_and_unsupported_scale(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.assertRaises(ValueError):
                round_away(value)
        with self.assertRaises(ValueError):
            scale_exponent(-1)
        with self.assertRaises(ValueError):
            quantize_dense([[1.0]], [0], 0, -20, False)
        with self.assertRaises(ValueError):
            quantize_dense([[1.0]], [1e20], 0, 0, False)
