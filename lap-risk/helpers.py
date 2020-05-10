# -*- coding: utf-8 -*-
import pickle, os, operator
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def drop_incomplete_cases(df: pd.DataFrame) -> (pd.DataFrame, int, int):
    """Drops incomplete rows in input DataFrame, printing pre- and
        post-drop summary stats."""
    total_n = df.shape[0]
    df = df.dropna()
    complete_n = df.shape[0]

    print(f'{total_n} cases in input DataFrame')
    print(f'Dropped {total_n - complete_n} incomplete cases',
          f'({np.round(100 * (1 - complete_n / total_n), 3)}%)')
    print(f'{complete_n} complete cases in returned Dataframe.')

    return df, total_n, complete_n


def split_into_folds(
    df: pd.DataFrame,
    tts_filepath: str = os.path.join('data', 'train_test_split.pkl')
) -> (pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray):
    """Splits supplied DataFrame into train and test folds, such that the test
        fold is the cases from the test trusts which are complete for the variables
        in the current NELA risk model, and the train fold is all the available
        cases from the train trusts. Two things to note:

        1) The train fold will be different between models as for the current NELA
        model it will only contain complete cases, whereas for our model it will
        also contain the cases that were incomplete prior to imputation.

        2) The test fold will be the same for all models, i.e. the current-NELA-
        model incomplete cases from the test fold trusts will not be used in
        training or testing of any of the models."""
    train_test_split = load_object(tts_filepath)
    split = {}

    for fold in ('train', 'test'):
        if fold == 'train':
            train_total = train_test_split[f'{fold}_i'].shape[0]
            train_test_split[f'{fold}_i'] = np.array([i for i in
                train_test_split[f'{fold}_i'] if i in df.index])
            train_intersection = train_test_split[f'{fold}_i'].shape[0]
            print(f'{train_total} cases in unabridged train fold.')
            print(f'Excluded {train_total - train_intersection} cases',
                  f'({np.round(100 * (1 - train_intersection / train_total), 3)}%)',
                  'not available in input DataFrame.')
            print(f'{train_intersection} cases in returned train Dataframe.')   

        split[fold] = {'X_df': df.loc[
            train_test_split[f'{fold}_i']].copy().reset_index(drop=True)}
        split[fold]['y'] = split[fold]['X_df']['Target'].values
        split[fold]['X_df'] = split[fold]['X_df'].drop('Target', axis=1)

    assert(split['test']['X_df'].shape[0] == train_test_split['test_i'].shape[0])

    return (split['train']['X_df'], split['train']['y'],
            split['test']['X_df'], split['test']['y'])


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


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)


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
