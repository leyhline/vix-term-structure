import unittest

from vixstructure.utils import parse_model_repr, parse_model_repr_old, parse_model_repr_monthwise


class TestUtils(unittest.TestCase):
    def test_parse_model_repr(self):
        result1 = parse_model_repr("20170703100041_tfpool08_depth9_width18_days1_dropout0e+00_optimAdam_lr1e-03")
        result2 = parse_model_repr("20170703100621_tfpool17_depth9_width21_days5_dropout0e+00_optimAdam_lr1e-03_normalized")
        self.assertEqual(len(result1), 9)
        self.assertEqual(len(result2), 9)
        self.assertEqual(result1.normalized, False)
        self.assertEqual(result2.normalized, True)
        
    def test_parse_model_repr_old(self):
        result1 = parse_model_repr_old("20170703100041_tfpool08_depth9_width18_dropout0e+00_optimAdam_lr1e-03")
        result2 = parse_model_repr_old("20170703100621_tfpool17_depth9_width21_dropout0e+00_optimAdam_lr1e-03_normalized")
        self.assertEqual(len(result1), 8)
        self.assertEqual(len(result2), 8)
        self.assertEqual(result1.normalized, False)
        self.assertEqual(result2.normalized, True)
        
    def test_parse_model_repr_monthwise(self):
        result1 = parse_model_repr_monthwise("20170816135917_tfpool45_depth9_width27_month10_dropout0e+00_optimAdam_lr1e-03")
        self.assertEqual(len(result1), 8)


if __name__ == '__main__':
    unittest.main()
