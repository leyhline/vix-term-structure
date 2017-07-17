import unittest

from vixstructure.utils import parse_model_repr


class TestUtils(unittest.TestCase):
    def test_parse_model_repr(self):
        result1 = parse_model_repr("20170703100041_tfpool08_depth9_width18_days1_dropout0e+00_optimAdam_lr1e-03")
        result2 = parse_model_repr("20170703100621_tfpool17_depth9_width21_days5_dropout0e+00_optimAdam_lr1e-03_normalized")
        self.assertEqual(len(result1), 9)
        self.assertEqual(len(result2), 9)
        self.assertEqual(result1.normalized, False)
        self.assertEqual(result2.normalized, True)


if __name__ == '__main__':
    unittest.main()
