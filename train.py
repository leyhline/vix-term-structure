# TODO Logging and Tests are essential!

import sys
import logging
from typing import Tuple, Iterator

import numpy as np
import keras

import vixstructure.models as models
import vixstructure.data as data





def get_model(hidden_layers, past_days, days_to_future, step_size=0.01):
    model = models.naive_fully_connected(hidden_layers, past_days, days_to_future)
    sgd = keras.optimizers.SGD(step_size)
    model.compile(sgd, keras.losses.mean_squared_error, metrics=['accuracy'])
    return model


def data_generator(data: np.ndarray, past_days: int, days_to_future: int
                   ) -> Iterator[np.ndarray]:
    # TODO Specify batch size.
    while True:  # Generator should loop indefinitely.
        for i in range(past_days, len(data) - days_to_future):
            # Expand dimension --> batch size of 1
            yield (np.expand_dims(data[i-past_days:i], axis=0),
                   np.expand_dims(data[i+days_to_future,1:], axis=0))


def get_data_generators(past_days: int, days_to_future: int,
             split: float=0.80, min_index: int=None, max_index: int=None,
             ) -> Tuple[
                    Tuple[int, Iterator[np.ndarray]],
                    Tuple[int, Iterator[np.ndarray]]]:
    """
    Get two generators, both which loop over their data indefinitely. The first one
    is for training, the second one for validation. Also returns the number of
    unique data samples until the generator starts the next loop.
    :param past_days:
    :param days_to_future:
    :param split: Fraction at which to split the data into training and test set.
    :param min_index: If you don't want to use the whole data (maybe because you also
                      need a test set) you can specify a range with ``min_index`` and
                      ``max_index``. Is ignored when greater than the data length.
    :param max_index: See ``min_index``.
    :return: A tuple of two tuples:
             1. tuple: (number of unique training samples, training data generator)
             2. tuple: (number of unique validation samples, validation data generator)
    """
    # TODO Think about a way to shuffle the data
    assert 0. < split < 1.
    if min_index: assert min_index >= 0
    if max_index: assert min_index < max_index
    training = data.get_data(normalized=True)
    # Check indixes.
    if min_index and min_index >= len(training):
        logging.warning(f"min_index is greater than length of data {len(training)}. Ignore.")
        min_index = None
    if max_index and max_index >= len(training):
        logging.warning(f"max_index is greater than length of data {len(training)}. Ignore.")
        max_index = None
    # Fill the NaN values and extract a numpy array.
    training = training.fillna(0).values[min_index:max_index]
    split_index = int(split * len(training))
    # Split data into validation set and training set.
    validation = training[split_index:]
    nr_samples_validation = len(validation) - past_days - days_to_future
    assert nr_samples_validation > 0
    training = training[:split_index]
    nr_samples_training = len(training) - past_days - days_to_future
    assert nr_samples_training > 0
    return ((nr_samples_training, data_generator(training, past_days, days_to_future)),
            (nr_samples_validation, data_generator(validation, past_days, days_to_future)))


def train(hidden_layers, past_days, days_to_future, epochs=100, verbose=1):
    repr_string = f"{hidden_layers}_{past_days}_{days_to_future}"
    (testlen, testgen), (vallen, valgen) = get_data_generators(past_days, days_to_future)
    model = get_model(hidden_layers, past_days, days_to_future)
    history = model.fit_generator(testgen, testlen, epochs=epochs, verbose=verbose,
                                  validation_data=valgen, validation_steps=vallen,
                                  callbacks=[keras.callbacks.CSVLogger(f"./logs/naive-fully-connected/training_{repr_string}.log"),
                                             keras.callbacks.TensorBoard("./logs/tensorboard")])
    model.save(f"./models/naive-fully-connected/naive_{repr_string}.hdf5")
    return model, history


def validate(model, past_days: int,
             min_index: int=-70, max_index: int=None):
    test_data = data.get_data(normalized=True).iloc[min_index:max_index].fillna(0).values
    max_index = 0 if not max_index else max_index
    test_batch = np.reshape(test_data, ((max_index - min_index) // past_days, past_days, 9))
    preds = model.predict(test_batch)
    return preds


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: train.py x y (where range(x,y) is the number of hidden layers)")
        print("Defaulting to 5 networks in range(1,6).")
        x = 1
        y = 6
    else:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
    print(f"Training {y-x} networks in range({x},{y}).")
    for i in range(x, y):
        print(f"Training with {i} hidden layer.")
        train(i, 7, 7, 100, 2)
