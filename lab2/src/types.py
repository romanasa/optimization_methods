import numpy as np
from plotly import graph_objects as go


class LinearRegression:
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def eval(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2
        X = np.hstack((X, np.ones(len(X)).reshape(-1, 1)))
        return X.dot(self.weights).reshape(-1, 1)

    def calc_mse_grad(self, x, y_true):
        y_pred = self.eval(x)
        x = np.hstack((x, np.ones(len(x)).reshape(-1, 1)))
        return np.mean(2 * (y_pred - y_true) * x, axis=0)

    @staticmethod
    def get_contour(x_train, y_train, min_c, max_c, cnt=10):
        assert x_train.shape[1] == 1

        linspace = np.linspace(min_c, max_c, cnt)

        criterion = MSE()

        z = []
        for y in linspace:
            z.append([])
            for x in linspace:
                model = LinearRegression(np.array([x, y]))
                y_pred = model.eval(x_train)
                z[-1].append(criterion.eval(y_pred, y_train))

        return go.Contour(
            z=z,
            x=linspace,
            y=linspace,
            contours_coloring='lines',
            line_width=2,
            contours=dict(
                start=0,
                end=10,
                size=1,
            ),
        )


class MSE:
    def eval(self, y_pred: np.ndarray, y_true: np.ndarray):
        return np.mean(np.square(y_pred - y_true))
