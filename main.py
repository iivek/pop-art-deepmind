import numpy as np
import torch
import time
import argparse

from utils.dataset_generator import generate_dataset
from learners.model import Model
from learners.normalized_sgd import NormalizedSGD
from learners.pop_art_sgd import PopArtSGD, ArtSGD
from learners.vanilla_sgd import VanillaSGD
from utils.utils import moving_average, median_and_percentile, save_results


# parser settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "-l", "--lr", default=-3.5, help="learning rate, default: 10^-3.5"
)
parser.add_argument(
    "-b", "--beta", default=0, help="moving average coefficient, default: None"
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    default="SGD",
    help="agent mode, default: SGD, one of ['SGD', 'ART', 'PopArt']",
)


def spawn_learner(lr, beta, mode):
    kwargs = {"model": Model(16, 10, 1), "lr": lr, "beta": beta}
    if mode == "sgd":
        learner = VanillaSGD(**kwargs)
    elif mode == "art":
        learner = ArtSGD(**kwargs)
    elif mode == "pop-art":
        learner = PopArtSGD(**kwargs)
    elif mode == "normalized-sgd":
        learner = NormalizedSGD(**kwargs)
    return learner


if __name__ == "__main__":

    args = parser.parse_args()
    lr = pow(10.0, float(args.lr))
    beta = pow(10.0, float(args.beta))
    mode = args.mode

    rmses = []
    for seed in range(50):
        print("Running seed {:d}.".format(seed))
        torch.manual_seed(seed)
        learner = spawn_learner(lr, beta, mode)
        start_tic = time.time()

        x, y = generate_dataset(seed)
        rmse = learner.train(x, y)
        print(
            "Time elapsed {:.2f} seconds for run {:d}.".format(
                time.time() - start_tic, seed
            )
        )
        rmses.append(rmse)

    # storing results
    samples = np.linspace(0, 4995, 4995, dtype=int)
    m, l, u = median_and_percentile(
        [moving_average(rmse) for rmse in rmses], axis=0
    )
    save_results(
        f"./results/{mode}_lr={args.lr}_beta={args.beta}.pkl", samples, m, l, u
    )
