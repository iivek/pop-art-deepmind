import numpy as np
import torch

from learners.losses import rmse_loss, mse_loss
from learners.model import Model


class History:
    """A lightweight collector of training history."""

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.history = []

    @classmethod
    def collect_history(cls, fun):
        def collect(self, inputs, outputs, *args, **kwargs):
            self.history.append(fun(self, inputs, outputs, *args, **kwargs))

        return collect


class Trainer(History):
    def __init__(self, model: Model, lr, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.loss = mse_loss
        self.opt_lower = torch.optim.SGD(
            self.model.lower_layers.parameters(), lr
        )
        self.opt_upper = torch.optim.SGD(
            self.model.upper_layer.parameters(), lr
        )

    def backward(self, loss):
        self.opt_lower.zero_grad()
        self.opt_upper.zero_grad()
        loss.backward()

    def predict(self, x):
        return self.model.forward(x)

    def step(self):
        self.opt_lower.step()
        self.opt_upper.step()

    def train_step(self, *args, **kwargs):
        pass

    @History.collect_history
    def train_metrics(self, inputs, outputs, *args, **kwargs):
        with torch.no_grad():
            metric = rmse_loss(self.predict(inputs), outputs)
        return metric.item()

    def train(self, x, y):
        """Train by taking a single datapoint per step"""
        for _x, _y in zip(x, y):
            self.train_step(
                inputs=torch.tensor(_x, dtype=torch.float),
                outputs=torch.tensor(_y, dtype=torch.float),
            )
            self.train_metrics(
                inputs=torch.tensor(_x, dtype=torch.float),
                outputs=torch.tensor(_y, dtype=torch.float),
            )
        return self.history


class PopArt:
    """Routines for tracking and updating statistics, by pop and art rescaling."""

    def __init__(self, *args, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = torch.tensor(1.0, dtype=torch.float)
        self.sigma_new = None
        self.mu = torch.tensor(0.0, dtype=torch.float)
        self.mu_new = None
        self.nu = self.sigma**2 + self.mu**2  # second-order moment
        self.beta = beta

    def art(self, y):
        self.mu_new = (1.0 - self.beta) * self.mu + self.beta * y
        self.nu = (1.0 - self.beta) * self.nu + self.beta * y**2
        self.sigma_new = np.sqrt(self.nu - self.mu_new**2)

    def pop(self, linear_layer):
        relative_sigma = self.sigma / self.sigma_new
        linear_layer.output_linear.weight.data.mul_(relative_sigma)
        linear_layer.output_linear.bias.data.mul_(relative_sigma).add_(
            (self.mu - self.mu_new) / self.sigma_new
        )

    def normalize(self, y):
        return (y - self.mu) / self.sigma

    def denormalize(self, y):
        return self.sigma * y + self.mu
