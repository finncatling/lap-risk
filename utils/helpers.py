import os
from typing import List

import numpy as np


def check_system_resources():
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    print('{:.1f} GB RAM'.format(mem_bytes / (1024 ** 3)))
    print('{} CPUs'.format(os.cpu_count()))


class GammaTransformer:
    """Transforms variable to more closely approximate
        (in the case of albumin) a gamma distribution.
        All that is required to fit the transformer
        is the winsor thresholds."""

    def __init__(self,
                 winsor_thresholds: List[float],
                 eps: float = 1e-16):
        self.low, self.high = winsor_thresholds
        self.eps = eps

    def transform(self, arr: np.ndarray):
        """Unlike sklearn, arr should be 1D, i.e. of
            shape (n_samples,). We add eps to remove
            zeros, as gamma is strictly positive."""
        return (self.high - arr) + self.eps

    def inverse_transform(self, arr: np.ndarray):
        """Unlike sklearn, arr should be 1D, i.e. of
            shape (n_samples,). Any (e.g. imputed)
            values outside the original winsor thresholds
            are winsorized."""
        arr = self.high - arr
        arr[np.where(arr > self.high)] = self.high
        arr[np.where(arr < self.low)] = self.low
        return arr
