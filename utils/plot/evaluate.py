from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from arviz.stats.density_utils import kde

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
    """Plot calibration curves from each train-test split."""
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


def plot_calibration_subplots(
    p: np.ndarray,
    calib_curves: Tuple[List[np.ndarray], List[np.ndarray]],
    model_names: Tuple[str, str],
    curve_transparency: float
) -> Tuple[Figure, Axes]:
    """Plot figure with 2 subplots, each containing calibration curves."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for ax_i, ax in enumerate(axes):
        for calib_curve in calib_curves[ax_i]:
            ax.plot(p, calib_curve, c="tab:blue", alpha=curve_transparency)
        ax.plot([0, 1], [0, 1], linestyle="dotted", c="black")
        ax.set(
            title=model_names[ax_i],
            xlabel="Predicted mortality risk",
            ylabel="Estimated true mortality risk",
            xlim=[0, 1],
            ylim=[0, 1],
        )
    fig.tight_layout()
    return fig, axes


def plot_example_risk_distributions(
    y_pred_samples: np.ndarray,
    patient_indices: Tuple[int, ...],
    kde_bandwidths: Tuple[float, ...]
) -> Tuple[Figure, Axes]:
    """Plot predicted risk distributions (and corresponding point estimates)
        for example patients. y_pred_samples is (n_sampled_risks, n_patients).
        kde_bandwidths should be same length as patient_indices."""
    fig, ax = plt.subplots()
    for i, j in enumerate(patient_indices):
        grid, pdf = kde(y_pred_samples[:, j], bw=kde_bandwidths[i])
        ax.fill_between(grid, pdf, alpha=0.4, label=f'Patient {i + 1}')
        if i:
            ax.axvline(np.median(y_pred_samples[:, j]), c='black', ls=':')
        else:
            ax.axvline(
                np.median(y_pred_samples[:, j]),
                c='black',
                ls=':',
                label='Point prediction'
            )

    ax.set_ylim(bottom=0)
    ax.set(
        xlim=(0, 1),
        xlabel='Predicted risk of death',
        ylabel='Probability density'
    )
    ax.legend()

    return fig, ax
