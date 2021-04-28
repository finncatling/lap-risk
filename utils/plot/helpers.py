import os
from typing import Tuple, Callable

import numpy as np
import pandas as pd


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
    extensions: Tuple[str, ...] = ("pdf",),
    dpi: Tuple = (None,),
    **plot_func_kwargs,
) -> None:
    """Wraps plotting function so figures are saved. output_filename should
        lack extension."""
    fig, _ = plot_func(*plot_func_args, **plot_func_kwargs)
    for ext, this_dpi in zip(extensions, dpi):
        fig.savefig(
            os.path.join(output_dir, f"{output_filename}.{ext}"),
            dpi=this_dpi,
            format=ext,
            bbox_inches="tight",
        )


def convert_creatinine_urea(df: pd.DataFrame) -> pd.DataFrame:
    """Converts creatinine and urea to US units.

    Creatinine (mmol/L) is converted to Creatinine (mg/dL)
    Urea (mmol/L) is converted to BUN (mg/dL)

    Args:
        df: Contains 'S03SerumCreatinine' and 'S03Urea' columns for conversion

    Returns:
        Data with converted (but not renamed) creatinine and urea columns
    """
    df = df.copy()
    df['S03SerumCreatinine'] /= 88.42
    df['S03Urea'] /= 0.357
    return df
