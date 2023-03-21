import collections
import copy
from typing import Tuple

import numpy as np
from plotly import graph_objects as go

from src import algorithms
from src import types


def calc_metrics(
        function: types.QuadraticFunction,
        algo: algorithms.GradientDescent,
        function_ind: int,
        iter_cnt: int,
) -> Tuple[dict, dict]:
    metrics = collections.defaultdict(list)
    for it in range(iter_cnt):
        np.random.seed(it * (function_ind + 1))

        cur_function = copy.deepcopy(function)
        cur_algo = copy.deepcopy(algo)

        point = cur_algo.optimize(cur_function)
        eval_info = cur_function.get_eval_info()

        metrics['steps'].append(len(cur_algo.history_points))
        metrics['evals'].append(eval_info[0])
        metrics['grads'].append(eval_info[1])

    return {k: f'{np.mean(v):.2f} ± {np.std(v):.2f}' for k, v in metrics.items()}, cur_algo.get_history_xy()


def run(
        function: types.QuadraticFunction,
        function_ind: int,
        algo: algorithms.GradientDescent,
        algo_name: str,
        iter_cnt: int,
):
    metrics, history_xy = calc_metrics(function, algo, function_ind, iter_cnt)

    draw_function = copy.deepcopy(function)

    fig = go.Figure(draw_function.get_contour(-20, 20, 100))
    fig.add_trace(go.Scatter(
        history_xy,
        mode='markers+lines',
        marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),
    ))

    title = f"Function {function_ind + 1}, Algo: {algo_name}, Steps: {metrics['steps']}<br>"
    title += f"Function evaluations: {metrics['evals']}<br>"
    title += f"Gradient calculations: {metrics['grads']}<br>"

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        font=dict(
            family="Courier New, monospace",
            size=11,
            color="RebeccaPurple"
        ),
        width=700, height=700, autosize=False
    )
    return metrics, fig
