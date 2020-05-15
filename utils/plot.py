from typing import Tuple

import numpy as np


def generate_ci_quantiles(cis: Tuple[float]) -> np.ndarray:
    """For a Tuple of % confidence intervals, e.g. (95, 50), generates the
        corresponding quantiles, e.g. (0.025, 0.25, 0.75, 0.975)."""
    quantiles = []
    for ci in cis:
        diff = (100 - ci) / (2 * 100)
        quantiles += [diff, 1 - diff]
    return np.array(sorted(quantiles))
