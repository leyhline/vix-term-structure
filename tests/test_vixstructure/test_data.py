import unittest
import datetime

import numpy as np

from vixstructure.data import Data


class MyTestCase(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
