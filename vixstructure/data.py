"""
vix-term-structure.data

This module is for loading the training data.
The data here is lazily loaded.

:copyright: (c) 2017 Thomas Leyh
"""

import os
import random
import logging
import operator
from typing import Tuple, Iterator, Union, Sequence

import pandas as pd
import numpy as np
import tensorflow.contrib.keras as keras
from lazy import lazy


# See analysis.ipynb
# Before this date there are too many NaN values.
FIRST_DATE = "2006-10-23"
LAST_DATE = "2017-05-11"
KWARGS = dict(usecols=range(1,10), dtype=np.float32, parse_dates=[0], header=0, index_col=0, na_values=0)


def split_indices(a: int, b: int, val_split=0.15, test_split=0.15):
    """
    Calculate the necessary indices for splitting a dataset into
    training, validation and test set in a diversified manner.
    :param a: First index.
    :param b: Last index.
    :param val_split: Float describing the fraction used for validation.
    :param test_split: Float describing the fraction used for testing.
    :return: A 7-tuple of integers with the following values:
             a: First index for training data
             val1: Starting index of first validation split
             test1: Starting index of first testset split
             data: Second index for training data
             val2: Starting index of second validation split
             test2: Starting index of second testset split
             b: Last index of the data
    """
    half = int((b - a) / 2)
    val_len = int(half * val_split)
    test_len = int(half * test_split)
    val1 = a + half - val_len - test_len
    test1 = a + half - test_len
    data = a + half
    val2 = b - val_len - test_len
    test2 = b - test_len
    return a, val1, test1, data, val2, test2, b


def split_dataset(dataset, splits: Sequence[int]):
    """
    Split dataset according to splits sequence. Use with split_indices function.
    :param dataset: The dataset to split.
    :param splits: A sequence holding the indices. Currently has to be of length 7.
    :return: Three splits of the initial dataset holding training, validation and test data.
    """
    indices = pd.DataFrame((split_indices(splits[i], splits[i+1]) for i in range(len(splits) - 1)))
    d1, v1, t1, d2, v2, t2 = ([dataset.iloc[a:b] for a, b in zip(indices.iloc[:,i], indices.iloc[:,i+1])]
                              for i in range(len(indices.columns) - 1))
    return pd.concat(d1 + d2), pd.concat(v1 + v2), pd.concat(t1 + t2)


class Data:
    def __init__(self, path, use_standard_kwargs=True,
                 first_index=FIRST_DATE, last_index=LAST_DATE, **kwargs):
        """
        Specify parameters for reading the data from path.
        :param path: The relative path to the data file. Should be a csv.
        :param use_standard_kwargs:
            Most of the files have the same structure and you can therefore use
            the global KWARGS. If True (default) then you can still overwrite these
            with the local kwargs argument.
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
            self.kwargs = KWARGS.copy()
            self.kwargs.update(kwargs)
        else:
            self.kwargs = kwargs
        self.first_index = first_index
        self.last_index = last_index

    @lazy
    def data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.filename, **self.kwargs).loc[self.first_index:self.last_index]


class VIX(Data):
    def __init__(self, path):
        super(VIX, self).__init__(path, use_standard_kwargs=False, parse_dates=[0],
                                  header=0, index_col=0, na_values="null", dtype=np.float32)

    @lazy
    def adjClose(self):
        return self.data_frame["Adj Close"]

    @lazy
    def normalized(self):
        return (self.adjClose - self.adjClose.mean()) / self.adjClose.std()


class Expirations(Data):
    def __init__(self, path):
        super(Expirations, self).__init__(path, dtype=None, parse_dates=list(range(0,9)))

    @lazy
    def for_first_leg(self):
        return self.data_frame["V1"]

    @lazy
    def days_to_expiration(self):
        until_expiration = pd.Series(self.for_first_leg.values - self.for_first_leg.index.values,
                                     index=self.for_first_leg.index, name="DtE")
        return until_expiration.map(operator.attrgetter("days")).astype(np.float32)


class TermStructure(Data):
    def __init__(self, path, expirations: Expirations=None):
        self.expirations = expirations
        super(TermStructure, self).__init__(path)

    @lazy
    def diff(self):
        diff = self.data_frame.diff(axis=1).iloc[:,1:]
        diff = diff.join(self.data_frame.iloc[:,0])
        return diff.iloc[:,range(-1,7)]

    @lazy
    def long_prices(self):
        assert self.expirations, "expirations need to be given in constructor"
        spreads = self.data_frame.copy()
        # First get the index of all dates where the expiration is in 0 days
        expiration_indices = self.expirations.days_to_expiration
        assert spreads.index.identical(expiration_indices.index)
        expiration_indices = expiration_indices.where(expiration_indices == 0.0)
        expiration_indices.index = range(len(expiration_indices))
        expiration_indices = expiration_indices.dropna().index
        # The day after a future expired the spread prices shift to the right
        if expiration_indices[-1] == len(spreads) - 1:
            shift_end = -1
        else:
            shift_end = None
        spreads.iloc[expiration_indices[:shift_end] + 1] = spreads.iloc[expiration_indices[:shift_end] + 1].iloc[:, 1:].assign(
            M1=lambda x: np.NaN).iloc[:, range(-1, 7)]
        assert len(spreads[spreads.M1.isnull() == True]) == len(expiration_indices) + 0 if not shift_end else shift_end
        spreads = spreads.apply(self._calculate_long_prices, axis=1)
        return spreads

    @staticmethod
    def _calculate_long_prices(term: pd.Series):
        """
        Calculate the spread prices for long positions for one term structure.
        :param term: A term structure as time series.
        :return: The spread prices for each leg except the ones on each side.
        """
        longs = [2*term[i] - term[i-1] - term[i+1] for i in range(1, len(term)-1)]
        return pd.Series(longs, term[1:-1].index)


class LongPricesDataset:
    def __init__(self, term_structure_path, expirations_path):
        self.expirations = Expirations(expirations_path)
        self.term_structure = TermStructure(term_structure_path, self.expirations)
        self.mean = dict()
        self.ptp = dict()

    def normalize_data(self, data: pd.DataFrame, idx) -> pd.DataFrame:
        """
        All values should be normalized to range(-1,1).
        :param data: The data to normalize.
        :param idx: An id for remembering normalization values in class.
        :return: Normalized DataFrame.
        """
        self.mean[idx] = data.mean()
        self.ptp[idx] = data.max() - data.min()
        return (data - self.mean[idx]) / self.ptp[idx]

    def denormalize_data(self, data: np.ndarray, idx) -> np.ndarray:
        return data * self.ptp[idx].values + self.mean[idx].values

    def denorm_mse(self, y_true, y_pred):
        """
        Calculate mean squared error on denormalized data.
        :param y_true: Target value.
        :param y_pred: Predicted value at output.
        :return: Mean squared error after denormalization.
        """
        # See keras/losses.py
        return keras.backend.mean(keras.backend.square(
            self.denormalize_data(y_pred, "y") - self.denormalize_data(y_true, "y")), axis=-1)

    def dataset(self, with_expirations=True, normalize=False,
                with_months=False, with_days=False,
                days_to_future=1) -> Tuple[np.ndarray, np.ndarray]:
        x = self.term_structure.diff
        y = self.term_structure.long_prices
        if with_months:
            x["month"] = x.index.month
        if with_days:
            x["day"] = x.index.day
        if with_expirations:
            x = x.join(self.expirations.days_to_expiration)
        if normalize:
            x = self.normalize_data(x, "x")
            y = self.normalize_data(y, "y")
        assert x.index.identical(y.index)
        return (x.iloc[:-days_to_future].fillna(0).values,
                y.iloc[days_to_future:].fillna(0).values)

    def splitted_dataset(self, validation_split: float=0.15, test_split: float=0.15,
                         with_expirations=True, normalize=False,
                         with_months=False, with_days=False,
                         days_to_future=1) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                    Tuple[np.ndarray, np.ndarray],
                                                    Tuple[np.ndarray, np.ndarray]]:
        """
        Split whole dataset into three parts: training set, cross validation set, test set.
        For the splits there are values used from the middle and from the end of the data to
        equal parts.
        :param validation_split: How large is the part used for validation?
        :param test_split: How large is the part used for testing?
        :param with_expirations: Passed to dataset method.
        :param normalize: Passed to dataset method.
        :return: Three (x,y)-tuples for the three above mentioned dataset splits.
        """
        x, y = self.dataset(with_expirations=with_expirations, normalize=normalize,
                            with_months=with_months, with_days=with_days, days_to_future=days_to_future)
        assert len(x) == len(y)
        val_length = int(len(x) * validation_split / 2)
        test_length = int(len(x) * test_split / 2)
        x_fst = x[:int(len(x) / 2)]
        x_snd = x[int(len(x) / 2):]
        y_fst = y[:int(len(y) / 2)]
        y_snd = y[int(len(y) / 2):]
        x_train, y_train = (np.append(x_fst[:-(val_length+test_length)], x_snd[:-(val_length+test_length)], axis=0),
                            np.append(y_fst[:-(val_length+test_length)], y_snd[:-(val_length+test_length)], axis=0))
        x_val, y_val = (np.append(x_fst[-(val_length+test_length):-test_length], x_snd[-(val_length+test_length):-test_length], axis=0),
                        np.append(y_fst[-(val_length+test_length):-test_length], y_snd[-(val_length+test_length):-test_length], axis=0))
        x_test, y_test = (np.append(x_fst[-test_length:], x_snd[-test_length:], axis=0),
                          np.append(y_fst[-test_length:], y_snd[-test_length:], axis=0))
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class VIXLongPrice:
    """
    Mapping long price to long price.
    This doesn't work very good.
    """
    def __init__(self, term_structure_path, expirations_path, vix_path):
        self.expirations = Expirations(expirations_path)
        self.term_structure = TermStructure(term_structure_path, self.expirations)
        self.vix = VIX(vix_path)

    def dataset(self, with_expirations=True, with_vix=False):
        x = self.term_structure.long_prices[:-1]
        y = self.term_structure.long_prices[1:]
        if with_vix:
            x = x.join(self.vix.adjClose)
        if with_expirations:
            x = x.join(self.expirations.days_to_expiration)
        return x.fillna(0).values, y.fillna(0).values

    def splitted_dataset(self, validation_split: float = 0.15, test_split: float = 0.15,
                         with_expirations=True, with_vix=False) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                                         Tuple[np.ndarray, np.ndarray],
                                                                         Tuple[np.ndarray, np.ndarray]]:
        x, y = self.dataset(with_expirations=with_expirations, with_vix=with_vix)
        assert len(x) == len(y)
        val_length = int(len(x) * validation_split / 2)
        test_length = int(len(x) * test_split / 2)
        x_fst = x[:int(len(x) / 2)]
        x_snd = x[int(len(x) / 2):]
        y_fst = y[:int(len(y) / 2)]
        y_snd = y[int(len(y) / 2):]
        x_train, y_train = (np.append(x_fst[:-(val_length + test_length)], x_snd[:-(val_length + test_length)], axis=0),
                            np.append(y_fst[:-(val_length + test_length)], y_snd[:-(val_length + test_length)], axis=0))
        x_val, y_val = (np.append(x_fst[-(val_length + test_length):-test_length], x_snd[-(val_length + test_length):-test_length],
                                  axis=0),
                        np.append(y_fst[-(val_length + test_length):-test_length], y_snd[-(val_length + test_length):-test_length],
                                  axis=0))
        x_test, y_test = (np.append(x_fst[-test_length:], x_snd[-test_length:], axis=0),
                          np.append(y_fst[-test_length:], y_snd[-test_length:], axis=0))
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class MinutelyData:
    """
    The same as LongPricesDataset but with minutely data.

    This class is very specific because the constructor assumes that there
    is one hdf5 file that contains all the data in a preprocessed way.
    """
    def __init__(self, hdf5_path):
        self.x = pd.read_hdf(hdf5_path, "x")
        self.y = pd.read_hdf(hdf5_path, "y")

    def dataset(self):
        return self.x.fillna(0).values, self.y.fillna(0).values

    def splitted_dataset(self, validation_split: float=0.15, test_split: float=0.15):
        x, y = self.dataset()
        assert len(x) == len(y)
        val_length = int(len(x) * validation_split / 2)
        test_length = int(len(x) * test_split / 2)
        x_fst = x[:int(len(x) / 2)]
        x_snd = x[int(len(x) / 2):]
        y_fst = y[:int(len(y) / 2)]
        y_snd = y[int(len(y) / 2):]
        x_train, y_train = (np.append(x_fst[:-(val_length+test_length)], x_snd[:-(val_length+test_length)], axis=0),
                            np.append(y_fst[:-(val_length+test_length)], y_snd[:-(val_length+test_length)], axis=0))
        x_val, y_val = (np.append(x_fst[-(val_length+test_length):-test_length], x_snd[-(val_length+test_length):-test_length], axis=0),
                        np.append(y_fst[-(val_length+test_length):-test_length], y_snd[-(val_length+test_length):-test_length], axis=0))
        x_test, y_test = (np.append(x_fst[-test_length:], x_snd[-test_length:], axis=0),
                          np.append(y_fst[-test_length:], y_snd[-test_length:], axis=0))
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class FutureswiseLongPrice:
    """
    Alternative representation of LongPricesDataset.
    Instead of mapping all legs of the term structure to all
    futures' spreads at once this maps to only a single month's spread.
    """
    def __init__(self, hdf5_path):
        self.x = pd.read_hdf(hdf5_path, "x")
        self.y = pd.read_hdf(hdf5_path, "y")

    def dataset(self):
        return self.x.fillna(0).values, self.y.fillna(0).values

    def splitted_dataset(self, validation_split: float=0.15, test_split: float=0.15):
        split_between_years = np.append(0, self.y.index.get_loc(str(self.y.index[-1])) + 1)
        x_train, x_val, x_test = split_dataset(self.x, split_between_years)
        y_train, y_val, y_test = split_dataset(self.y, split_between_years)
        x_train = x_train.fillna(0).values
        y_train = y_train.fillna(0).values
        x_val = x_val.fillna(0).values
        y_val = y_val.fillna(0).values
        x_test = x_test.fillna(0).values
        y_test = y_test.fillna(0).values
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class FuturesByMonth:
    """
    Another representation of long prices and the respective term structure.
    With these data frames one can select futures by month (or by year but
    that's rather silly).
    """
    def __init__(self, hdf5_path):
        self.x = pd.read_hdf(hdf5_path, "x")
        self.y = pd.read_hdf(hdf5_path, "y")

    def dataset(self, month: int, diff=False):
        """
        Select a mapping x-y pair for a specific month.
        :param month: Integer value between 1 and 12.
        :param diff: Get delta/difference over input values.
        :return: Tuple with input data and target data.
        """
        x = self.x.copy()
        x = x.loc(axis=0)[:, month]
        if diff:
            x = x.diff(axis=1).iloc[:, 1:]
        return x.fillna(0).values, self.y.loc(axis=0)[:, month].fillna(0).values

    def splitted_dataset(self, month: int, validation_split: float=0.15, test_split: float=0.15,
                         diff=False):
        x, y = self.dataset(month, diff=diff)
        assert len(x) == len(y)
        val_length = int(len(x) * validation_split / 2)
        test_length = int(len(x) * test_split / 2)
        x_fst = x[:int(len(x) / 2)]
        x_snd = x[int(len(x) / 2):]
        y_fst = y[:int(len(y) / 2)]
        y_snd = y[int(len(y) / 2):]
        x_train, y_train = (np.append(x_fst[:-(val_length + test_length)], x_snd[:-(val_length + test_length)], axis=0),
                            np.append(y_fst[:-(val_length + test_length)], y_snd[:-(val_length + test_length)], axis=0))
        x_val, y_val = (np.append(x_fst[-(val_length + test_length):-test_length], x_snd[-(val_length + test_length):-test_length], axis=0),
                        np.append(y_fst[-(val_length + test_length):-test_length], y_snd[-(val_length + test_length):-test_length], axis=0))
        x_test, y_test = (np.append(x_fst[-test_length:], x_snd[-test_length:], axis=0),
                          np.append(y_fst[-test_length:], y_snd[-test_length:], axis=0))
        y_train = np.expand_dims(y_train, axis=1)
        y_val = np.expand_dims(y_val, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


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
