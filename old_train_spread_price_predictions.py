#!/usr/bin/env python3

import sys
import socket

import tensorflow.contrib.keras as keras
import pandas as pd
import numpy as np

import vixstructure.models as models


def get_model(hidden_layers, step_size, dropout=None):
    model = models.spread_price_prediction(hidden_layers, 12, dropout)
    sgd = keras.optimizers.SGD(step_size)
    model.compile(sgd, keras.losses.mean_squared_error, metrics=['accuracy'])
    return model


def normalize(data: pd.DataFrame):
    mean = data.mean()
    ptp = data.max() - data.min()
    return (data - mean) / ptp


def train(hidden_layers, dropout=None, epochs=50, verbose=1, validation_split=0.8, sgd_step_size=0.01):
    lines_without_useful_data = 650
    term_structure = pd.read_csv("data/annual_structure.csv", header=0, index_col=0, dtype=np.float32, parse_dates=[0],
                                 skiprows=lines_without_useful_data)
    spread_prices = pd.read_csv("data/long_prices.csv", header=0, index_col=0, dtype=np.float32, parse_dates=[0],
                                skiprows=lines_without_useful_data)
    assert len(term_structure) == len(spread_prices)
    term_structure = normalize(term_structure)
    spread_prices = normalize(spread_prices)
    term_structure = term_structure.fillna(1.)
    spread_prices = spread_prices.fillna(1.)
    splitsize = int(len(term_structure) * validation_split)
    x_train = term_structure.values[:splitsize]
    y_train = spread_prices.values[:splitsize]
    x_val = term_structure.values[splitsize:]
    y_val = spread_prices.values[splitsize:]
    model = get_model(hidden_layers, sgd_step_size, dropout)
    repr_string = "training_{}_{}_{}_{}".format(hidden_layers, "nodropout" if not dropout else dropout, sgd_step_size, socket.gethostname())
    history = model.fit(x_train, y_train, 1, epochs=epochs, verbose=verbose, validation_data=(x_val, y_val), shuffle=True,
                        callbacks=[keras.callbacks.CSVLogger("./logs/predict_spread_prices/{}.csv".format(repr_string))])
    model.save("./models/predict_spread_prices/{}.hdf5".format(repr_string))


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: train.py x y (where range(x,y) is the number of hidden layers)")
    #     print("Defaulting to 5 networks in range(1,6).")
    #     x = 1
    #     y = 6
    # else:
    #     x = int(sys.argv[1])
    #     y = int(sys.argv[2])
    # print(f"Training {y-x} networks in range({x},{y}).")
    # for i in range(x, y):
    #     print(f"Training with {i} hidden layer.")
    #     train(i)
    for layer in range(8, 11):
        for dropout in (None, 0.5):
            for stepsize in (0.01, 0.003, 0.001, 0.0003, 0.0001):
                print("Train model with {} hidden layers, dropout is {}, stepsize {}.".format(layer, dropout, stepsize))
                train(layer, dropout=dropout, epochs=100, verbose=0, sgd_step_size=stepsize)
