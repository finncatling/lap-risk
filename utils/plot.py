import os
from typing import Tuple, Dict, List, Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def generate_ci_quantiles(cis: Tuple[float]) -> np.ndarray:
    """For a Tuple of % confidence intervals, e.g. (95, 50), generates the
        corresponding quantiles, e.g. (0.025, 0.25, 0.75, 0.975)."""
    quantiles = []
    for ci in cis:
        diff = (100 - ci) / (2 * 100)
        quantiles += [diff, 1 - diff]
    return np.array(sorted(quantiles))


def plot_saver(plot_func: Callable,
               *plot_func_args,
               output_dir: str,
               output_filename: str,
               extensions: Tuple[str] = ('pdf', 'eps'),
               **plot_func_kwargs) -> None:
    """Wraps plotting function so figures are saved. output_filename should
        lack extension."""
    fig, _ = plot_func(*plot_func_args, **plot_func_kwargs)
    for ext in extensions:
        fig.savefig(os.path.join(output_dir, f'{output_filename}.{ext}'),
                    format=ext, bbox_inches='tight')


def plot_stratified_risk_distributions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        hist_args: Dict = {'bins': 50, 'density': True, 'alpha': 0.5,
                           'range': (0, 1)}
) -> Tuple[Figure, Axes]:
    """Plots predicted risks, stratified by mortality label."""
    fig, ax = plt.subplots()
    for i, outcome in enumerate(('Alive', 'Dead')):
        stratified_y_pred = y_pred[np.where(y_true == i)[0]]
        ax.hist(stratified_y_pred, label=outcome, **hist_args)
    ax.set(xlabel='Predicted mortality risk', ylabel='Probability density')
    ax.legend()
    return fig, ax


def plot_calibration(p: np.ndarray,
                     calib_curves: List[np.ndarray],
                     curve_transparency: float) -> Tuple[Figure, Axes]:
    """Plot calibration curve, with confidence intervals."""
    fig, ax = plt.subplots(figsize=(4, 4))
    for calib_curve in calib_curves:
        ax.plot(p, calib_curve, c='tab:blue', alpha=curve_transparency)
    ax.plot([0, 1], [0, 1], linestyle='dotted', c='black')
    ax.set(xlabel='Predicted mortality risk',
           ylabel='Estimated true mortality risk',
           xlim=[0, 1], ylim=[0, 1])
    return fig, ax
