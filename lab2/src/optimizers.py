import abc
import time

import numpy as np

from src import types


class Base:
    def __init__(self, lr: float, decay: float, clip: float = 10):
        self.lr = lr
        self.decay = decay
        self.history = []
        self.clip = clip

        self.time_sum = 0
        self.time_cnt = 0

    def get_new_params(self, point, grad):
        grad = np.clip(grad, -self.clip, self.clip)
        self.history.append(point)
        return self.get_next_point(point, grad)

    def step(self, x: np.ndarray, y: np.ndarray, model: types.LinearRegression):
        start_time = time.time_ns()

        self._step(x, y, model)

        self.time_sum += (time.time_ns() - start_time)
        self.time_cnt += 1

    def get_history(self):
        history = np.array(self.history)
        return {
            'x': history[:, 0],
            'y': history[:, 1],
        }

    def _step(self, x: np.ndarray, y: np.ndarray, model: types.LinearRegression):
        grad = model.calc_mse_grad(x, y)
        model.weights = self.get_new_params(model.weights, grad)

    def decay_lr(self):
        self.lr *= self.decay

    @abc.abstractmethod
    def get_next_point(self, point, grad):
        pass


class Standard(Base):
    def __init__(self, lr: float, decay: float = 1):
        super().__init__(lr, decay)

    def get_next_point(self, point, grad):
        return point - self.lr * grad


class Momentum(Base):
    def __init__(self, lr: float, momentum: float, decay: float = 1):
        super().__init__(lr, decay)
        self.momentum = momentum
        self.velocity = None

    def get_next_point(self, point, grad):
        if self.velocity is None:
            self.velocity = np.zeros_like(grad)
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return point - self.velocity


class Nesterov(Base):
    def __init__(self, lr: float, momentum: float, decay: float = 1):
        super().__init__(lr, decay)
        self.momentum = momentum
        self.velocity = None

    def _step(self, x: np.ndarray, y: np.ndarray, model: types.LinearRegression):
        if self.velocity is not None:
            model_hat = types.LinearRegression(model.weights - self.momentum * self.velocity)
            grad = model_hat.calc_mse_grad(x, y)
        else:
            grad = model.calc_mse_grad(x, y)
        model.weights = self.get_new_params(model.weights, grad)

    def get_next_point(self, point, grad):
        if self.velocity is None:
            self.velocity = np.zeros_like(grad)
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return point - self.velocity


class AdaGrad(Base):
    def __init__(self, lr: float, decay: float = 1):
        super().__init__(lr, decay)
        self.eps = 1e-10
        self.r = None

    def get_next_point(self, point, grad):
        if self.r is None:
            self.r = np.zeros_like(grad)
        self.r += np.square(grad)
        lr = self.lr / (np.sqrt(self.r) + self.eps)
        return point - lr * grad


class RmsProp(Base):
    def __init__(self, lr: float, p: float = 0.99, decay: float = 1):
        super().__init__(lr, decay)
        self.p = p
        self.r = None
        self.eps = 1e-10

    def get_next_point(self, point, grad):
        if self.r is None:
            self.r = np.zeros_like(grad)
        self.r = self.p * self.r + (1 - self.p) * np.square(grad)

        lr = self.lr / (np.sqrt(self.r) + self.eps)
        return point - lr * grad


class Adam(Base):
    def __init__(self, lr: float, betta1: float = 0.9, betta2: float = 0.999, decay: float = 1):
        super().__init__(lr, decay)
        self.betta1 = betta1
        self.betta2 = betta2

        self.v = None
        self.r = None
        self.eps = 1e-10
        self.t = 0

    def get_next_point(self, point, grad):
        if self.v is None:
            self.v = np.zeros_like(grad)
            self.r = np.zeros_like(grad)

        self.t += 1

        self.v = self.betta1 * self.v + (1 - self.betta1) * grad
        self.r = self.betta2 * self.r + (1 - self.betta2) * np.square(grad)

        v = self.v / (1 - np.power(self.betta1, self.t))
        r = self.r / (1 - np.power(self.betta2, self.t))

        lr = self.lr / (np.sqrt(r) + self.eps)
        return point - lr * v
