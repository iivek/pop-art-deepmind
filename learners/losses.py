import torch


def rmse_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def mse_loss(yhat, y):
    return 0.5 * torch.mean((yhat - y) ** 2)
