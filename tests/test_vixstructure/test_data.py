import unittest
import datetime

from vixstructure.data import Data


class MyTestCase(unittest.TestCase):
    def test_if_data_frame_for_settle_data_is_complete(self):
        settle = Data("../../data/8_m_settle.csv")
        df = settle.data_frame
        first = df.iloc[0]
        self.assertEqual(first.name.date(), datetime.date(2006, 10, 23))
        self.assertEqual(len(df), 2656)
        self.assertEqual(df.isnull().sum().sum(), 621)


if __name__ == '__main__':
    unittest.main()
