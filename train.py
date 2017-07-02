import argparse

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


def train(args):
    model = models.term_structure_to_spread_price(args.network_depth, args.network_width, args.dropout)
    print(model.summary())
    optimizer = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
    dataset = data.LongPricesDataset("data/8_m_settle.csv", "data/expirations.csv")
    (x_train, y_train), (x_val, y_val), _ = dataset.splitted_dataset(normalize=args.normalize)
    metrics = []
    if args.normalize:
        metrics.append(dataset.denorm_mse)
    model.compile(optimizer, "mean_squared_error", metrics=metrics)
    model.fit(x_train, y_train, args.batch_size, args.epochs, verbose=0 if args.quiet else 2,
              validation_data=(x_val, y_val))


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
