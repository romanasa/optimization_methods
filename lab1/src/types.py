import numpy as np
from plotly import graph_objects as go


# x.T @ A @ x + b.T @ x + c
class QuadraticFunction:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: float):
        assert len(A.shape) == 2
        assert len(b.shape) == 1
        assert A.shape[0] == A.shape[1] == b.shape[0]
        self.A = A
        self.b = b
        self.c = c

        self.eval_cnt = 0
        self.grad_cnt = 0

    @classmethod
    def generate_random(cls, n: int, k: int) -> 'QuadraticFunction':
        a = np.random.uniform(1, k, n)
        b = np.random.randn(n)
        a[0] = 1
        a[-1] = k
        return cls(np.diag(a), b, 0)

    def eval(self, x: np.ndarray):
        self.eval_cnt += 1
        assert len(x.shape) == 1
        assert x.shape[0] == self.A.shape[0]
        sum_2 = np.dot(np.dot(x.T, self.A), x)
        sum_1 = np.dot(self.b.T, x)
        return sum_2 + sum_1 + self.c

    def calc_gradient(self, x: np.ndarray):
        self.grad_cnt += 1
        assert len(x.shape) == 1
        assert x.shape[0] == self.A.shape[0]
        grad_a = np.dot((self.A + self.A.T), x)
        grad_b = self.b
        return grad_a + grad_b

    def get_eval_info(self):
        return self.eval_cnt, self.grad_cnt

    def get_shape(self):
        return self.b.shape[0]

    def get_scale(self):
        scale = np.maximum(np.max(self.A, axis=0), np.max(self.A, axis=1))
        scale = np.maximum(scale, self.b)
        return 1 / scale

    def get_contour(self, min_c, max_c, cnt=10):
        assert self.get_shape() == 2
        linspace = np.linspace(min_c, max_c, cnt)
        z = [[self.eval(np.array([x, y])) for x in linspace] for y in linspace]
        return go.Contour(
            z=np.log2(z),
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
