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
parser.add_argument("-d", "--dropout", type=float, default=None)
parser.add_argument("-op", "--optimizer", choices=["Adam", "SGD"], default="Adam")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-bs", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-n", "--normalize", action="store_true")
parser.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("-s", "--save", type=str, help="Optional: Specify save path for trained model.")
parser.add_argument("--reduce_lr", action="store_true", help="If validation loss stagnates, reduce lr by sqrt(0.1).")
parser.add_argument("--shuffle_off", action="store_false", help="Don't shuffle training data.")


def train(args):
    model = models.term_structure_to_spread_price(args.network_depth, args.network_width, args.dropout)
    optimizer = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
    dataset = data.LongPricesDataset("data/8_m_settle.csv", "data/expirations.csv")
    (x_train, y_train), (x_val, y_val), _ = dataset.splitted_dataset(normalize=args.normalize)
    metrics = []
    if args.normalize:
        metrics.append(dataset.denorm_mse)
    model.compile(optimizer, "mean_squared_error", metrics=metrics)
    callbacks = []
    if args.save:
        now = datetime.datetime.now()
        name = "{}_{}_depth{}_width{}_dropout{:.0e}_optim{}_lr{:.0e}{}".format(
                now.strftime("%Y%m%d%H%M%S"),
                socket.gethostname(),
                args.network_depth,
                args.network_width,
                0 if not args.dropout else args.dropout,
                args.optimizer,
                args.learning_rate,
                "_normalized" if args.normalize else "")
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

