"""
CLI for training feedforward neural networks with prepared data.
Second data representation after first approach failed.
"""

import argparse
import sys
import os
import datetime
import socket
from math import sqrt

import tensorflow.contrib.keras as keras

import vixstructure.data as data
import vixstructure.models as models

parser = argparse.ArgumentParser(description="Train a fully-connected neural network.")
parser.add_argument("network_depth", type=int)
parser.add_argument("network_width", type=int)
parser.add_argument("month", type=int)
parser.add_argument("-d", "--dropout", type=float, default=None)
parser.add_argument("-op", "--optimizer", choices=["Adam", "SGD"], default="Adam")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-bs", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("-s", "--save", type=str, help="Optional: Specify save path for trained model.")
parser.add_argument("-a", "--activation", default="relu", help="Activation function for hidden layers.",
                    choices=["relu", "selu"])
parser.add_argument("--reduce_lr", action="store_true", help="If validation loss stagnates, reduce lr by sqrt(0.1).")
parser.add_argument("--shuffle_off", action="store_false", help="Don't shuffle training data.")
parser.add_argument("--yearly", action="store_true", help="Inputs x now always have 12 rows.")
parser.add_argument("--diff", action="store_true", help="Take the delta of input before training.")
parser.add_argument("--spreads", action="store_true", help="Use input to calculate spread prices before training.")


def train(args):
    # Check if month number is valid.
    if args.month not in range(1, 13):
        print("Month argument has to be an integer between 1 and 12.", file=sys.stderr)
        sys.exit(1)
    if args.yearly:
        input_data_length = 12
    else:
        if args.spreads:
            input_data_length = 6
        else:
            input_data_length = 8
    model = models.term_structure_to_single_spread_price(args.network_depth, args.network_width,
                                                         args.dropout, input_data_length, args.activation)
    optimizer = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
    dataset = data.FuturesByMonth("data/futures_per_year_and_month.h5", yearly=args.yearly)
    (x_train, y_train), (x_val, y_val), _ = dataset.splitted_dataset(args.month, diff=args.diff, spreads=args.spreads)
    model.compile(optimizer, "mean_squared_error")
    callbacks = []
    if args.save:
        now = datetime.datetime.now()
        name = "{}_{}_depth{}_width{}_month{}_dropout{:.0e}_optim{}_lr{:.0e}".format(
            now.strftime("%Y%m%d%H%M%S"),
            socket.gethostname(),
            args.network_depth,
            args.network_width,
            args.month,
            0 if not args.dropout else args.dropout,
            args.optimizer,
            args.learning_rate)
        callbacks.append(keras.callbacks.CSVLogger(os.path.join(args.save, name + ".csv")))
    if args.reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(factor=sqrt(0.1), patience=100, verbose=1))
    model.fit(x_train, y_train, args.batch_size, args.epochs, verbose=0 if args.quiet else 2,
              validation_data=(x_val, y_val), callbacks=callbacks, shuffle=args.shuffle_off)
    if args.save:
        try:
            model.save_weights(os.path.join(args.save, name + ".h5"))
        except FileNotFoundError as e:
            print("Could not save the model.", str(e), file=sys.stderr)


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
