import json
from pathlib import Path
import tempfile
import unittest

from models.quantization.dense_reference import DenseSpec, export_model, fixture, requantize


class ArithmeticTests(unittest.TestCase):
    def test_rounding_ties_and_neighbors(self):
        for value, expected in [(0, 0), (1, 1), (3, 2), (-1, -1), (-3, -2)]:
            self.assertEqual(requantize(value, 1), expected)
        for value, expected in [(1, 0), (2, 1), (3, 1), (-1, 0), (-2, -1), (-3, -1)]:
            self.assertEqual(requantize(value, 2), expected)

    def test_saturation_and_extreme_accumulators(self):
        for value, expected in [(128, 127), (-129, -128), (127, 127), (-128, -128)]:
            self.assertEqual(requantize(value, 0), expected)
        self.assertEqual(requantize(-(1 << 31), 31), -1)
        self.assertEqual(requantize((1 << 31) - 1, 31), 1)
        self.assertEqual(requantize(-(1 << 31), 0), -128)
        self.assertEqual(requantize(-999, 0, relu=True), 0)

    def test_hand_calculated_dense(self):
        model = DenseSpec(((2, -3), (-1, 4)), (1, -2), shift=1)
        self.assertEqual(model.evaluate([5, -2]), ([17, -15], [9, -8]))

    def test_worst_case_products(self):
        model = DenseSpec(((-128,) * 8, (127,) * 8), (0, 0), shift=0)
        self.assertEqual(model.evaluate([-128] * 8), ([131072, -130048], [127, -128]))

    def test_invalid_parameters(self):
        for kwargs in [dict(weights=(), biases=()),
                       dict(weights=((1,), (1, 2)), biases=(0, 0)),
                       dict(weights=((128,),), biases=(0,)),
                       dict(weights=((1,),), biases=()),
                       dict(weights=((1,),), biases=(0,), shift=-1),
                       dict(weights=((1,),), biases=(0,), shift=32),
                       dict(weights=((1,),), biases=(0,), relu=1),
                       dict(weights=((1,),), biases=((1 << 31) - 1,)),
                       dict(weights=((-128, -128),), biases=(0,), acc_bits=16)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DenseSpec(**kwargs)

    def test_invalid_inputs(self):
        model = DenseSpec(((1,),), (0,))
        for values in [[], [1, 2], [128], [-129], [1.0], [True]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.evaluate(values)

    def test_export_layout_and_roundtrip(self):
        model = DenseSpec(((-128, 127), (1, -1)), (-123, 456), shift=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            export_model(model, path)
            self.assertEqual((path / "weights.mem").read_text(), "80\n7f\n01\nff\n")
            self.assertEqual((path / "biases.mem").read_text(), "ffffff85\n000001c8\n")
            restored = DenseSpec.from_dict(json.loads((path / "model.json").read_text()))
            self.assertEqual(restored, model)

    def test_reject_mismatched_metadata(self):
        model, _ = fixture()
        for key, value in [("inputs", 9), ("rounding", "floor"), ("schema_version", 2)]:
            data = model.to_dict()
            data[key] = value
            with self.assertRaises(ValueError):
                DenseSpec.from_dict(data)

    def test_fixture_is_reproducible(self):
        self.assertEqual(fixture(42), fixture(42))
        self.assertNotEqual(fixture(42), fixture(43))


if __name__ == "__main__":
    unittest.main()
