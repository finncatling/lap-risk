import operator
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def winsorize(df: pd.DataFrame,
              thresholds_dict: Dict[str, Tuple[float, float]] = None,
              cont_vars: List[str] = None,
              quantiles: Tuple[float, float] = (0.001, 0.999),
              include: Dict[str, Tuple[bool, bool]] = None) -> (
        pd.DataFrame, Dict[str, Tuple[float, float]]):
    """Winsorize continuous variables at thresholds in
        thresholds_dict, or at specified quantiles if thresholds_dict
        is None. If thresholds_dict is None, upper and/or lower
        Winsorization for selected variables can be disabled using the
        include dict. Variables not specified in the include dict have
        Winsorization applied at upper and lower thresholds by
        default."""
    df = df.copy()

    ops = (operator.lt, operator.gt)

    if thresholds_dict:
        for v, thresholds in thresholds_dict.items():
            for i, threshold in enumerate(thresholds):
                if threshold is not None:
                    df.loc[ops[i](df[v], threshold), v] = threshold
    else:
        thresholds_dict = {}
        for v in cont_vars:
            thresholds_dict[v] = list(df[v].quantile(quantiles))
            for i, threshold in enumerate(thresholds_dict[v]):
                try:
                    if include[v][i]:
                        df.loc[ops[i](df[v], threshold), v] = threshold
                    else:
                        thresholds_dict[v][i] = None
                except KeyError:
                    df.loc[ops[i](df[v], threshold), v] = threshold

    return df, thresholds_dict


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
