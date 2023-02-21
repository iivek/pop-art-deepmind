"""Adapted from https://github.com/zouyu4524/Pop-Art-Translation"""

import numpy as np
import pickle
import argparse


def dec2bin(i, n=16):
    b = np.binary_repr(i, n)
    return [int(s) for s in b]


def generate_dataset(seed):
    np.random.seed(seed)

    y_range = np.linspace(0, 1023, 1024, dtype=int)
    y = np.random.choice(y_range, 5000, replace=True)

    y_weird = 65535
    y = y.reshape((1000, 5))
    y = np.asarray(
        y.tolist() + (np.ones((1, 5)) * y_weird).tolist(), dtype=int
    )
    y = y.reshape(-1, order="F")
    y = y[:-1]

    x = np.asarray([dec2bin(target, n=16) for target in y])

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="random seed to generate dataset, default: 0",
    )
    args = parser.parse_args()
    seed = args.seed
    x, y = generate_dataset(seed)

    with open("./dataset.pkl", "wb") as f:
        pickle.dump(x, f)
        pickle.dump(y, f)
