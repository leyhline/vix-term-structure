# TODO Logging and Tests are essential!

import sys

import numpy as np
import keras

import vixstructure.models as models
import vixstructure.data as data


def get_model(hidden_layers, past_days, days_to_future, step_size=0.01):
    model = models.naive_fully_connected(hidden_layers, past_days, days_to_future)
    sgd = keras.optimizers.SGD(step_size)
    model.compile(sgd, keras.losses.mean_squared_error, metrics=['accuracy'])
    return model


def train(hidden_layers, past_days, days_to_future, epochs=100, verbose=1):
    repr_string = f"{hidden_layers}_{past_days}_{days_to_future}"
    (testlen, testgen), (vallen, valgen) = data.get_data_generators(past_days, days_to_future)
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
