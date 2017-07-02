import unittest

from vixstructure.models import term_structure_to_spread_price


class TestModels(unittest.TestCase):
    def test_term_structure_to_spread_price(self):
        model = term_structure_to_spread_price(5, 9)
        self.assertEqual(len(model.layers), 7)


if __name__ == '__main__':
    unittest.main()
