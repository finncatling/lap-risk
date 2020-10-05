from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.evaluate import stratify_y_pred


def plot_stratified_risk_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hist_bins: int = 50,
    hist_density: bool = True,
    hist_alpha: float = 0.5,
    hist_range: Tuple[float] = (0, 1)
) -> Tuple[Figure, Axes]:
    """Plots predicted risks, stratified by mortality label."""
    fig, ax = plt.subplots()
    stratified_y_pred = stratify_y_pred(y_true, y_pred)
    for i, outcome in enumerate(("Alive", "Dead")):
        ax.hist(
            stratified_y_pred[i],
            label=outcome,
            bins=hist_bins,
            density=hist_density,
            alpha=hist_alpha,
            range=hist_range
        )
    ax.set(xlabel="Predicted mortality risk", ylabel="Probability density")
    ax.legend()
    return fig, ax


def plot_calibration(
    p: np.ndarray,
    calib_curves: List[np.ndarray],
    curve_transparency: float
) -> Tuple[Figure, Axes]:
    """Plot calibration curve, with confidence intervals."""
    fig, ax = plt.subplots(figsize=(4, 4))
    for calib_curve in calib_curves:
        ax.plot(p, calib_curve, c="tab:blue", alpha=curve_transparency)
    ax.plot([0, 1], [0, 1], linestyle="dotted", c="black")
    ax.set(
        xlabel="Predicted mortality risk",
        ylabel="Estimated true mortality risk",
        xlim=[0, 1],
        ylim=[0, 1],
    )
    return fig, ax
