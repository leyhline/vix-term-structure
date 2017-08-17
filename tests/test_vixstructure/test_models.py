import unittest

import tensorflow.contrib.keras as keras
import numpy as np

from vixstructure.models import term_structure_to_spread_price, term_structure_to_spread_price_v2
from vixstructure.models import mask_output
from vixstructure.data import LongPricesDataset


class TestModels(unittest.TestCase):
    def setUp(self):
        self.dataset = LongPricesDataset("../../data/8_m_settle.csv", "../../data/expirations.csv")

    def test_term_structure_to_spread_price(self):
        model = term_structure_to_spread_price(5, 9)
        self.assertEqual(len(model.layers), 7)

    def test_mask_output_function_for_lambda_layers(self):
        input = keras.layers.Input(shape=(9,))
        output = keras.layers.Lambda(mask_output)(input)
        model = keras.models.Model(inputs=input, outputs=output)
        x, y = self.dataset.dataset()
        preds = model.predict(x)
        self.assertEqual(preds.shape, (2655, 6))
        self.assertEqual(np.all(preds, axis=0).sum(), 5)
        self.assertEqual(np.all(preds, axis=1).sum(), 2529)
        self.assertEqual((preds == 0.).sum(), 126)

    def test_term_structure_to_spread_prices_v2(self):
        model = term_structure_to_spread_price_v2(5, 9)
        x, y = self.dataset.dataset()
        preds = model.predict(x)
        self.assertEqual(preds.shape, (2655, 6))
        self.assertEqual(np.all(preds, axis=0).sum(), 5)
        self.assertEqual(np.all(preds, axis=1).sum(), 2529)


if __name__ == '__main__':
    unittest.main()
