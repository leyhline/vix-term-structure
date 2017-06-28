"""
vix-term-structure.data

This module is for loading the training data.
The data here is lazily loaded.

:copyright: (c) 2017 Thomas Leyh
"""

import os
import random
import logging
from typing import Tuple, Iterator, Union

import pandas as pd
import numpy as np
from lazy import lazy


# See analysis.ipynb
# Before this date there are too many NaN values.
FIRST_DATE = "2006-10-23"
KWARGS = dict(usecols=range(1,10), dtype=np.float32, parse_dates=[0], header=0, index_col=0, na_values=0)


class Data:
    def __init__(self, path, use_standard_kwargs=True, first_index=FIRST_DATE, **kwargs):
        """
        Specify parameters for reading the data from path.
        :param path: The relative path to the data file. Should be a csv.
        :param use_standard_kwargs:
            Most of the files have the same structure and you can therefore use
            the global KWARGS. If True (default) then the given kwargs are overwritten.
        :param first_index:
            The first index where reading the data makes sense because often there is
            too much noise (speak: NaN) at the beginning. This is a DataFrame index and
            therefore doesn't have to be an integer. In fact, most of the time it's a date.
        :param kwargs:
            These are passed to pandas.read_csv.
        """
        assert os.path.exists(path)
        self.filename = path
        if use_standard_kwargs:
            kwargs.update(KWARGS)
        self.kwargs = kwargs
        self.first_index = first_index

    @lazy
    def data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.filename, **self.kwargs).loc[self.first_index:]


################################################################
# All this is old stuff. Better use classes for holding data.
################################################################


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


def data_generator(data: np.ndarray, past_days: int, days_to_future: int,
                   shuffle: bool=False
                   ) -> Iterator[np.ndarray]:
    # TODO Specify batch size.
    index_range = list(range(past_days, len(data) - days_to_future))
    while True:  # Generator should loop indefinitely.
        if shuffle:
            random.shuffle(index_range)
        for i in index_range:
            # Expand dimension --> batch size of 1
            yield (np.expand_dims(data[i-past_days:i], axis=0),
                   np.expand_dims(data[i+days_to_future,1:], axis=0))


def get_data_generators(past_days: int, days_to_future: int,
                        split: Union[None,float]=0.80,
                        min_index: int=None, max_index: int=None,
                        ) -> Tuple[Tuple[int, Iterator[np.ndarray]],
                                   Tuple[int, Iterator[np.ndarray]]]:
    """
    Get two generators, both which loop over their data indefinitely. The first one
    is for training, the second one for validation. Also returns the number of
    unique data samples until the generator starts the next loop.
    :param past_days:
    :param days_to_future:
    :param split: Fraction at which to split the data into training and test set.
                  Must be a float in range (0,1). If None then there is only one
                  generator filled with all available data.
    :param min_index: If you don't want to use the whole data (maybe because you also
                      need a test set) you can specify a range with ``min_index`` and
                      ``max_index``. Is ignored when greater than the data length.
    :param max_index: See ``min_index``.
    :return: A tuple of two tuples:
             1. tuple: (number of unique training samples, training data generator)
             2. tuple: (number of unique validation samples, validation data generator)
    """
    if min_index:
        assert min_index >= 0
    if max_index:
        assert min_index < max_index
    training = get_data(normalized=True)
    # Check indixes.
    if min_index and min_index >= len(training):
        logging.warning("min_index is greater than length of data {}. Ignore.".format(len(training)))
        min_index = None
    if max_index and max_index >= len(training):
        logging.warning("max_index is greater than length of data {}. Ignore.".format(len(training)))
        max_index = None
    # Fill the NaN values and extract a numpy array.
    training = training.fillna(0).values[min_index:max_index]
    if not split:
        nr_samples = len(training) - past_days - days_to_future
        return nr_samples, data_generator(training, past_days, days_to_future)
    assert 0. < split < 1.
    split_index = int(split * len(training))
    # Split data into validation set and training set.
    validation = training[split_index:]
    nr_samples_validation = len(validation) - past_days - days_to_future
    assert nr_samples_validation > 0
    training = training[:split_index]
    nr_samples_training = len(training) - past_days - days_to_future
    assert nr_samples_training > 0
    return ((nr_samples_training, data_generator(training, past_days, days_to_future, shuffle=True)),
            (nr_samples_validation, data_generator(validation, past_days, days_to_future)))
