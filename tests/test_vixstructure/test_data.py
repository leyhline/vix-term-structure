import unittest
import datetime

import numpy as np

from vixstructure.data import Data, TermStructure, Expirations
from vixstructure.data import long_prices_dataset


class TestData(unittest.TestCase):
    def test_if_data_frame_for_settle_data_is_complete(self):
        settle = Data("../../data/8_m_settle.csv")
        df = settle.data_frame
        first = df.iloc[0]
        self.assertEqual(first.name.date(), datetime.date(2006, 10, 23))
        self.assertEqual(len(df), 2656)
        self.assertEqual(df.isnull().sum().sum(), 621)

    def test_if_data_frame_for_expirations_is_complete(self):
        expirations = Data("../../data/expirations.csv", dtype=None, parse_dates=list(range(0,9)))
        df = expirations.data_frame
        first = df.iloc[0]
        self.assertEqual(first.name.date(), datetime.date(2006, 10, 23))
        self.assertEqual(len(df), 2656)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_if_data_frame_for_vix_is_complete(self):
        vix = Data("../../data/vix.csv", last_index="2017-05-11", use_standard_kwargs=False,
                   parse_dates=[0], header=0, index_col=0, na_values="null", dtype=np.float32)
        df = vix.data_frame
        first = df.iloc[0]
        self.assertEqual(first.name.date(), datetime.date(2006, 10, 23))
        self.assertEqual(len(df), 2656)


class TestTermStructure(unittest.TestCase):
    def test_settle_diff(self):
        ts = TermStructure("../../data/8_m_settle.csv")
        diff = ts.diff
        self.assertEqual(diff.shape, (2656, 8))
        self.assertTrue((diff.iloc[:,1:].mean() < 1).all())

    def test_long_prices(self):
        expirations = Expirations("../../data/expirations.csv")
        ts = TermStructure("../../data/8_m_settle.csv", expirations)
        spreads = ts.long_prices
        self.assertEqual(spreads.shape, (2656, 6))
        self.assertTrue((spreads.min() > -14).all())
        self.assertTrue((spreads.max() < 5).all())
        self.assertEqual(spreads["M2"].isnull().sum(), 126)


class TestExpirations(unittest.TestCase):
    def test_for_first_leg(self):
        expirations = Expirations("../../data/expirations.csv")
        first_leg = expirations.for_first_leg
        self.assertEqual(first_leg.shape, (2656,))
        self.assertTrue(first_leg.dtype.type is np.datetime64)

    def test_days_to_expiration(self):
        expirations = Expirations("../../data/expirations.csv")
        to_expiration = expirations.days_to_expiration
        self.assertEqual(to_expiration.dtype.type, np.float32)
        self.assertEqual(to_expiration.shape, (2656,))
        self.assertGreaterEqual(to_expiration.min(), 0.0)
        self.assertLessEqual(to_expiration.max(), 34.0)


class TestDatasets(unittest.TestCase):
    def test_long_prices_dataset(self):
        x, y = long_prices_dataset("../../data/8_m_settle.csv", "../../data/expirations.csv")
        self.assertEqual(x.shape, (2655, 9))
        self.assertEqual(y.shape, (2655, 6))


if __name__ == '__main__':
    unittest.main()
