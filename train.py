import argparse

import tensorflow.contrib.keras as keras

import vixstructure.data as data
import vixstructure.models as models


parser = argparse.ArgumentParser(description="Train a fully-connected neural network.")
parser.add_argument("hidden_layers", type=int)
parser.add_argument("-d", "--dropout", type=float, default=None)
parser.add_argument("-op", "--optimizer", choices=["Adam", "SGD"], default="Adam")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-bs", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-vs", "--validation_split", type=float, default=0.2)
parser.add_argument("-n", "--normalize", action="store_true")
parser.add_argument("-q", "--quiet", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    model = models.term_structure_to_spread_price(args.hidden_layers, args.dropout)
    optimizer = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
    dataset = data.LongPricesDataset("data/8_m_settle.csv", "data/expirations.csv")
    x, y = dataset.dataset(normalize=args.normalize)
    metrics = []
    if args.normalize:
        metrics.append(dataset.denorm_mse)
    model.compile(optimizer, "mean_squared_error", metrics=metrics)
    model.fit(x, y, args.batch_size, args.epochs, verbose=0 if args.quiet else 2,
              validation_split=args.validation_split)
