import unittest
import datetime
from collections import Counter

import numpy as np
import pandas as pd

from vixstructure.data import Data, TermStructure, Expirations, VIX
from vixstructure.data import LongPricesDataset, VIXLongPrice, MinutelyData, FutureswiseLongPrice


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
    def setUp(self):
        self.ts = TermStructure("../../data/8_m_settle.csv")
        self.expirations = Expirations("../../data/expirations.csv")

    def test_settle_diff(self):
        diff = self.ts.diff
        self.assertEqual(diff.shape, (2656, 8))
        self.assertTrue((diff.iloc[:,1:].mean() < 1).all())

    def test_long_prices(self):
        self.ts = TermStructure("../../data/8_m_settle.csv", self.expirations)
        spreads = self.ts.long_prices
        self.assertEqual(spreads.shape, (2656, 6))
        self.assertTrue((spreads.min() > -14).all())
        self.assertTrue((spreads.max() < 5).all())
        self.assertEqual(spreads["M2"].isnull().sum(), 126)


class TestExpirations(unittest.TestCase):
    def setUp(self):
        self.expirations = Expirations("../../data/expirations.csv")

    def test_for_first_leg(self):
        first_leg = self.expirations.for_first_leg
        self.assertEqual(first_leg.shape, (2656,))
        self.assertTrue(first_leg.dtype.type is np.datetime64)

    def test_days_to_expiration(self):
        to_expiration = self.expirations.days_to_expiration
        self.assertEqual(to_expiration.dtype.type, np.float32)
        self.assertEqual(to_expiration.shape, (2656,))
        self.assertGreaterEqual(to_expiration.min(), 0.0)
        self.assertLessEqual(to_expiration.max(), 34.0)


class TestLongPricesDatasets(unittest.TestCase):
    def setUp(self):
        self.dataset = LongPricesDataset("../../data/8_m_settle.csv", "../../data/expirations.csv")

    def test_long_prices_dataset(self):
        x, y = self.dataset.dataset()
        self.assertEqual(x.shape, (2655, 9))
        self.assertEqual(y.shape, (2655, 6))
        x, y = self.dataset.dataset(with_expirations=False)
        self.assertEqual(x.shape, (2655, 8))
        self.assertEqual(y.shape, (2655, 6))

    def test_data_normalization(self):
        x_norm, y_norm = self.dataset.dataset(normalize=True)
        self.assertEqual(x_norm.shape, (2655, 9))
        self.assertEqual(y_norm.shape, (2655, 6))
        self.assertGreater(x_norm.min().min(), -1)
        self.assertLess(x_norm.max().max(), 1)
        self.assertGreater(y_norm.min().min(), -1)
        self.assertLess(y_norm.max().max(), 1)
        x = self.dataset.denormalize_data(x_norm, "x")
        y = self.dataset.denormalize_data(y_norm, "y")
        self.assertEqual(x.shape, (2655, 9))
        self.assertEqual(y.shape, (2655, 6))
        self.assertGreater(x.min().min(), -22)
        self.assertLess(x.max().max(), 70)
        self.assertGreater(y.min().min(), -14)
        self.assertLess(y.max().max(), 5)

    def test_dataset_split_into_train_test_and_validation_data(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.dataset.splitted_dataset()
        x_train_fst = x_train[:int(len(x_train) / 2)]
        x_train_snd = x_train[int(len(x_train) / 2):]
        x_val_fst = x_val[:int(len(x_val) / 2)]
        x_val_snd = x_val[int(len(x_val) / 2):]
        x_test_fst = x_test[:int(len(x_test) / 2)]
        x_test_snd = x_test[int(len(x_test) / 2):]
        x_full = np.concatenate([x_train_fst, x_val_fst, x_test_fst, x_train_snd, x_val_snd, x_test_snd], axis=0)
        x, y = self.dataset.dataset()
        self.assertTrue((x==x_full).all())
        y_train_fst = y_train[:int(len(y_train) / 2)]
        y_train_snd = y_train[int(len(y_train) / 2):]
        y_val_fst = y_val[:int(len(y_val) / 2)]
        y_val_snd = y_val[int(len(y_val) / 2):]
        y_test_fst = y_test[:int(len(y_test) / 2)]
        y_test_snd = y_test[int(len(y_test) / 2):]
        y_full = np.concatenate([y_train_fst, y_val_fst, y_test_fst, y_train_snd, y_val_snd, y_test_snd], axis=0)
        self.assertTrue((y==y_full).all())

    def test_if_month_and_day_gets_included(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.dataset.splitted_dataset(with_months=True,
                                                                                             with_days=True)
        for data in (x_train, x_val, x_test):
            self.assertEqual(data.shape[1], 11)

    def test_five_days_prediction(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.dataset.splitted_dataset(days_to_future=5)
        x_train_fst = x_train[:int(len(x_train) / 2)]
        x_train_snd = x_train[int(len(x_train) / 2):]
        x_val_fst = x_val[:int(len(x_val) / 2)]
        x_val_snd = x_val[int(len(x_val) / 2):]
        x_test_fst = x_test[:int(len(x_test) / 2)]
        x_test_snd = x_test[int(len(x_test) / 2):]
        x_full = np.concatenate([x_train_fst, x_val_fst, x_test_fst, x_train_snd, x_val_snd, x_test_snd], axis=0)
        x, y = self.dataset.dataset(days_to_future=5)
        self.assertEqual(x.shape, (2651, 9))
        self.assertEqual(y.shape, (2651, 6))
        self.assertTrue((x == x_full).all())
        y_train_fst = y_train[:int(len(y_train) / 2)]
        y_train_snd = y_train[int(len(y_train) / 2):]
        y_val_fst = y_val[:int(len(y_val) / 2)]
        y_val_snd = y_val[int(len(y_val) / 2):]
        y_test_fst = y_test[:int(len(y_test) / 2)]
        y_test_snd = y_test[int(len(y_test) / 2):]
        y_full = np.concatenate([y_train_fst, y_val_fst, y_test_fst, y_train_snd, y_val_snd, y_test_snd], axis=0)
        self.assertTrue((y == y_full).all())


class TestVIX(unittest.TestCase):
    def setUp(self):
        self.vix = VIX("../../data/vix.csv")

    def test_adjclose(self):
        self.assertEqual(self.vix.adjClose.shape, (2656,))
        self.assertAlmostEqual(self.vix.adjClose.mean(), 20.2477, places=4)


class TestVIXLongPrice(unittest.TestCase):
    def setUp(self):
        self.dataset = VIXLongPrice("../../data/8_m_settle.csv", "../../data/expirations.csv", "../../data/vix.csv")

    def test_basic_dataset(self):
        x, y = self.dataset.dataset(with_expirations=False, with_vix=False)
        self.assertEqual(x.shape, (2655, 6))
        self.assertEqual(y.shape, (2655, 6))

    def test_extended_dataset(self):
        x, y = self.dataset.dataset(with_expirations=True, with_vix=True)
        self.assertEqual(x.shape, (2655, 8))
        self.assertEqual(y.shape, (2655, 6))

    def test_dataset_split_into_train_test_and_validation_data(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.dataset.splitted_dataset(with_vix=True)
        x_train_fst = x_train[:int(len(x_train) / 2)]
        x_train_snd = x_train[int(len(x_train) / 2):]
        x_val_fst = x_val[:int(len(x_val) / 2)]
        x_val_snd = x_val[int(len(x_val) / 2):]
        x_test_fst = x_test[:int(len(x_test) / 2)]
        x_test_snd = x_test[int(len(x_test) / 2):]
        x_full = np.concatenate([x_train_fst, x_val_fst, x_test_fst, x_train_snd, x_val_snd, x_test_snd], axis=0)
        x, y = self.dataset.dataset(with_vix=True)
        self.assertTrue((x==x_full).all())
        y_train_fst = y_train[:int(len(y_train) / 2)]
        y_train_snd = y_train[int(len(y_train) / 2):]
        y_val_fst = y_val[:int(len(y_val) / 2)]
        y_val_snd = y_val[int(len(y_val) / 2):]
        y_test_fst = y_test[:int(len(y_test) / 2)]
        y_test_snd = y_test[int(len(y_test) / 2):]
        y_full = np.concatenate([y_train_fst, y_val_fst, y_test_fst, y_train_snd, y_val_snd, y_test_snd], axis=0)
        self.assertTrue((y==y_full).all())


class TestMinutelyData(unittest.TestCase):
    def setUp(self):
        self.dataset = MinutelyData("../../data/term_structure_minutely.h5")

    def test_basic_dataset(self):
        x, y = self.dataset.dataset()
        self.assertEqual(x.shape, (651600, 9))
        self.assertEqual(y.shape, (651600, 6))

    def test_dataset_split_into_train_test_and_validation_data(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.dataset.splitted_dataset()
        x_train_fst = x_train[:int(len(x_train) / 2)]
        x_train_snd = x_train[int(len(x_train) / 2):]
        x_val_fst = x_val[:int(len(x_val) / 2)]
        x_val_snd = x_val[int(len(x_val) / 2):]
        x_test_fst = x_test[:int(len(x_test) / 2)]
        x_test_snd = x_test[int(len(x_test) / 2):]
        x_full = np.concatenate([x_train_fst, x_val_fst, x_test_fst, x_train_snd, x_val_snd, x_test_snd], axis=0)
        x, y = self.dataset.dataset()
        self.assertTrue((x==x_full).all())
        y_train_fst = y_train[:int(len(y_train) / 2)]
        y_train_snd = y_train[int(len(y_train) / 2):]
        y_val_fst = y_val[:int(len(y_val) / 2)]
        y_val_snd = y_val[int(len(y_val) / 2):]
        y_test_fst = y_test[:int(len(y_test) / 2)]
        y_test_snd = y_test[int(len(y_test) / 2):]
        y_full = np.concatenate([y_train_fst, y_val_fst, y_test_fst, y_train_snd, y_val_snd, y_test_snd], axis=0)
        self.assertTrue((y==y_full).all())


class TestFutureswiseLongPrice(unittest.TestCase):
    def setUp(self):
        self.dataset = FutureswiseLongPrice("../../data/futureswise_mapping.h5")

    def test_data_from_constructor(self):
        c1 = Counter(self.dataset.y.index - self.dataset.x.index)
        c2 = Counter({pd.Timedelta(days=1): 11882,
                      pd.Timedelta(days=2): 139,
                      pd.Timedelta(days=3): 2762,
                      pd.Timedelta(days=4): 400,
                      pd.Timedelta(days=5): 10})
        self.assertTrue(c1 == c2)

    def test_dataset(self):
        x, y = self.dataset.dataset()
        self.assertEqual(x.shape, (15193, 8))
        self.assertEqual(y.shape, (15193,))

    def test_splitted_dataset(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.dataset.splitted_dataset()
        x = np.concatenate([x_train, x_val, x_test], axis=0)
        y = np.concatenate([y_train, y_val, y_test], axis=0)
        self.assertEqual(x.shape, (15193, 8))
        self.assertEqual(y.shape, (15193,))


if __name__ == '__main__':
    unittest.main()
