"""
vix-term-structure.data

This module is for loading the training data.
The data here is lazily loaded.

:copyright: (c) 2017 Thomas Leyh
"""

import os

import pandas as pd
import numpy as np


SETTLE_PATH = "../data/8_m_settle.csv"
VIX_PATH = "../data/vix.csv"

# Internal variables, use getter functions.
_MEAN = None
_PTP = None
_DATA = None


def get_data(normalized=False):
    global _DATA
    if _DATA is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xm_settle = pd.read_csv(os.path.join(current_dir, SETTLE_PATH),
                                usecols=range(1, 10), dtype=np.float32,
                                parse_dates=[0], header=0, index_col=0, na_values=0)
        vix = pd.read_csv(os.path.join(current_dir, VIX_PATH),
                            usecols=[0,5], parse_dates=[0], header=0, index_col=0,
                            na_values=["null", 0], dtype = np.float32)
        _DATA = pd.merge(vix, xm_settle, left_index=True, right_index=True)
    if normalized:
        return normalize(_DATA)
    else:
        return _DATA


def get_mean():
    global _MEAN
    if _MEAN is None:
        _MEAN = get_data().mean()
    return _MEAN


def get_ptp():
    """ptp = peak to peak"""
    global _PTP
    if _PTP is None:
        _PTP = get_data().max() - get_data().min()
    return _PTP


def normalize(data):
    """
    Normalize the given data using the mean and ptp of the imported training data.
    :param data: The unnormalized data.
    :return: All the data should now be in a [-1,1] interval.
    """
    return (data - get_mean()) / get_ptp()


def denormalize(data):
    """
    Inverts the normalize function above.
    :param data: Data which got normalized by the function above.
    :return: Denormalized original data.
    """
    return data * get_ptp()[1:].values + get_mean()[1:].values
