import os
import re
from typing import Tuple, Callable

import numpy as np

from utils.model.novel import INDICATION_PREFIX


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


def sanitize_indication(ind: str, ind_prefix: str = INDICATION_PREFIX) -> str:
    ind = ind[len(ind_prefix):]
    ind = ' '.join(re.findall('[A-Z][^A-Z]*', ind))
    ind = ind.lower()
    return ind[0].upper() + ind[1:]
