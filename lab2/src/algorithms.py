import time

import numpy as np
import torch
from torch.utils import data

from src import optimizers
from src.types import LinearRegression, MSE


def generate_points(n: int, d: int):
    regression = LinearRegression(np.arange(d + 1) + 2)
    x = np.random.randn(n, d)
    y = regression.eval(x)
    return x, y


def run_gradient_descent(X_train, y_train, batch_size: int, optimizer: optimizers.Base,
                         num_iters: int, eps: float):
    dataset = data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = data.DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    d = X_train.shape[1] + 1
    init_weights = np.random.uniform(-1 / (2 * d), 1 / (2 * d), d)
    model = LinearRegression(init_weights)

    criterion = MSE()

    loss_history = []

    ind = 0
    for epoch in range(num_iters):
        for x_train, y_true in dataloader:
            x_train, y_true = x_train.numpy(), y_true.numpy()

            y_pred = model.eval(x_train)

            loss = criterion.eval(y_pred, y_true)
            loss_history.append(loss)

            optimizer.step(x_train, y_true, model)

            ind += 1
            if ind == num_iters or loss < eps:
                return loss_history

        optimizer.decay_lr()
    return loss_history


def multiple_run(X_train, y_train, batch_size: int, optimizer: optimizers.Base,
                 num_iters: int, eps: float, return_history: bool = False,
                 multiple_cnt: int = 30):
    iters = []
    for it in range(multiple_cnt):
        np.random.seed(it)
        loss_history = run_gradient_descent(X_train, y_train, batch_size, optimizer, num_iters, eps)
        iters.append(len(loss_history))
    if return_history:
        return np.mean(iters), np.std(iters), loss_history
    return np.mean(iters), np.std(iters)
