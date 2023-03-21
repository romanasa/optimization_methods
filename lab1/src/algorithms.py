import abc
import math

import numpy as np

from src.types import QuadraticFunction


class GradientDescent:
    def __init__(self, start_mode: str, clip=10, **params):
        assert start_mode in ['zero', 'random', 'fixed']
        if start_mode == 'random':
            self.random_scale = params['random_scale']
        elif start_mode == 'fixed':
            self.start_point = params['start_point']
        self.start_mode = start_mode
        self.clip = clip
        self.epsilon = params['epsilon']
        self.use_scale = params['use_scale']
        self.history_points = []

    @abc.abstractmethod
    def get_alpha(self, function) -> float:
        pass

    def step(self, function: QuadraticFunction, point: np.ndarray, scale: np.ndarray) -> np.ndarray:
        assert function.get_shape() == point.shape[0]
        grad = -function.calc_gradient(point)
        grad *= scale
        grad[grad > self.clip] = self.clip
        grad[grad < -self.clip] = -self.clip

        alpha = self.get_alpha(lambda x: function.eval(point + x * grad))
        new_point = point + alpha * grad
        return new_point

    def get_start_point(self, n: int) -> np.ndarray:
        if self.start_mode == 'zero':
            return np.zeros(n)
        elif self.start_mode == 'random':
            return np.random.randn(n) * self.random_scale
        elif self.start_mode == 'fixed':
            return self.start_point

    def get_history_xy(self) -> dict:
        np_history = np.array(self.history_points)
        return {
            'x': np_history[:, 0],
            'y': np_history[:, 1],
        }

    def optimize(self, function: QuadraticFunction) -> np.ndarray:
        if self.use_scale:
            scale = function.get_scale()
        else:
            scale = np.ones(function.get_shape())

        point = self.get_start_point(function.get_shape())
        self.history_points.append(point)
        for num_iters in range(10_000):
            new_point = self.step(function, point, scale)
            distance = np.sqrt(np.square(new_point - point).sum())
            point = new_point
            self.history_points.append(point)
            if distance < self.epsilon:
                break
        return point


class ConstantGradientDescent(GradientDescent):
    def __init__(self, start_mode: str, alpha: float, **params):
        super().__init__(start_mode, **params)
        self.alpha = alpha

    def get_alpha(self, function) -> float:
        return self.alpha


class GoldenSearchGradientDescent(GradientDescent):
    def __init__(self, start_mode: str, max_alpha: float, eps: float, **params):
        super().__init__(start_mode, **params)
        self.max_alpha = max_alpha
        self.eps = eps

    def get_alpha(self, function) -> float:
        GOLDEN = (math.sqrt(5) - 1) / 2
        a = 0
        b = self.max_alpha
        x1, x2 = b * (1 - GOLDEN), b * GOLDEN
        f1, f2 = function(x1), function(x2)

        num_iters = 0
        while b - a > self.eps and num_iters < 100:
            if f1 < f2:
                b = x2
                x2, f2 = x1, f1
                x1 = b - GOLDEN * (b - a)
                f1 = function(x1)
            else:
                a = x1
                x1, f1 = x2, f2
                x2 = a + GOLDEN * (b - a)
                f2 = function(x2)
            num_iters += 1
        return a
