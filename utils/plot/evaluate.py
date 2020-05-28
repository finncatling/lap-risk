from typing import Dict, Any, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_stratified_risk_distributions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        hist_args: Dict[str, Any] = {'bins': 50, 'density': True, 'alpha': 0.5,
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
