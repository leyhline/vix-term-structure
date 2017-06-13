"""
vix-term-structure.models

In this file I will define some models (starting with some very simple ones)
to see what captures the existing data the best.

:copyright: (c) 2017 Thomas Leyh
"""

from typing import Optional

import tensorflow.contrib.keras as keras


def naive_fully_connected(hidden_layers: int, past_days: int, days_to_future: int):
    """
    This is a simple network consisting of a variable number of fully connected layers.
    It doesn't produce the final output (investment recommendations) but just tries to
    generate a future term structure. 
    
    "Isn't this useless" you might ask?
    
    Well, at the moment I still lack some data and expertise but this naive approach seems
    to be a good idea to investigate the temporal dependencies of the data in general.
    The question I want to answer here is: How many days do I have to look into the past
    to get a glimpse of the future?
    
    Though the main problem here is will be certainly overfitting. There is not much data,
    its not augmented and the model itself if because of its verbosity prone to overfitting.
    I'll try to counter this with dropout regularization.
    
    :param hidden_layers: Number of hidden layers.
    :param past_days: How many days to look into the past.
    :param days_to_future: For which day in the future to make the prediction.
    :return: A Keras model with these inputs (one vector for each day):
                - The VIX indices of ``past_days``
                - The term structures of ``past_days``
             The output is a term structure in ``days_to_future``.
             The model is not yet compiled.
    """
    initializer = keras.initializers.glorot_normal()
    activation = keras.activations.tanh
    input = keras.layers.Input(shape=(past_days, 9), name="input")
    hidden = keras.layers.Flatten()(input)
    for _ in range(hidden_layers):
        hidden = keras.layers.Dense(9 * past_days, activation=activation, kernel_initializer=initializer)(hidden)
        hidden = keras.layers.Dropout(rate=0.5)(hidden)
    output = keras.layers.Dense(8, activation=activation, name="output", kernel_initializer=initializer)(hidden)
    model = keras.models.Model(inputs=input, outputs=output)
    return model


def spread_price_prediction(hidden_layers: int, data_length: int, dropout: Optional[float]):
    initializer = keras.initializers.glorot_uniform()
    activation = keras.activations.relu
    input = keras.layers.Input(shape=(data_length,), name="input")
    hidden = input
    for _ in range(hidden_layers):
        hidden = keras.layers.Dense(data_length, activation=activation, kernel_initializer=initializer)(hidden)
        if dropout:
            hidden = keras.layers.Dropout(rate=dropout)(hidden)
    output = keras.layers.Dense(data_length, activation=activation, name="output", kernel_initializer=initializer)(hidden)
    model = keras.models.Model(inputs=input, outputs=output)
    return model
