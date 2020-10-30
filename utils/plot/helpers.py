import os
import re
from typing import Tuple, Callable

import numpy as np
from matplotlib.axes import Axes

from utils.indications import INDICATION_PREFIX


def generate_ci_quantiles(cis: Tuple[float]) -> np.ndarray:
    """For a Tuple of % confidence intervals, e.g. (95, 50), generates the
        corresponding quantiles, e.g. (0.025, 0.25, 0.75, 0.975)."""
    quantiles = []
    for ci in cis:
        diff = (100 - ci) / (2 * 100)
        quantiles += [diff, 1 - diff]
    return np.array(sorted(quantiles))


def plot_saver(
    plot_func: Callable,
    *plot_func_args,
    output_dir: str,
    output_filename: str,
    extensions: Tuple[str] = ("pdf", ),
    **plot_func_kwargs,
) -> None:
    """Wraps plotting function so figures are saved. output_filename should
        lack extension."""
    fig, _ = plot_func(*plot_func_args, **plot_func_kwargs)
    for ext in extensions:
        fig.savefig(
            os.path.join(output_dir, f"{output_filename}.{ext}"),
            format=ext,
            bbox_inches="tight",
        )


def sanitize_indication(ind: str, ind_prefix: str = INDICATION_PREFIX) -> str:
    ind = ind[len(ind_prefix):]
    ind = "\n".join(re.findall("[A-Z][^A-Z]*", ind))
    ind = ind.lower()
    return ind[0].upper() + ind[1:]


def autoscale_x(ax: Axes):
    """Hack for rescaling ax's x axis based on the data visible within the
        current xlim. Adapted from https://tinyurl.com/y6mz3eub """
    def get_low_high(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        bottom, top = ax.get_ylim()
        x_displayed = xd[((yd > bottom) & (yd < top))]
        low = np.min(x_displayed)
        high = np.max(x_displayed)
        return low, high

    lines = ax.get_lines()
    low, high = np.inf, -np.inf

    for line in lines:
        new_low, new_high = get_low_high(line)
        if new_low < low:
            low = new_low
        if new_high > high:
            high = new_high

    ax.set_xlim(low, high)
