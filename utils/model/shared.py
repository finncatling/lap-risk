from typing import Dict, List
import numpy as np
import pandas as pd
from utils.split_data import TrainTestSplitter, split_into_folds


def flatten_model_var_dict(model_vars: Dict) -> List[str]:
    """Flattens model variable name dict into single list. Function
        placed in this file to avoid cyclical dependencies."""
    return (list(model_vars['cat']) + list(model_vars['cont']) +
            [model_vars['target']])


class Splitter:
    """Base class to handle repeated train-test splitting, according to
        pre-defined splits in passed TrainTestSplitter."""
    def __init__(self,
                 df: pd.DataFrame,
                 test_train_splitter: TrainTestSplitter,
                 target_variable_name: str):
        self.df = df
        self.tts = test_train_splitter
        self.target_variable_name = target_variable_name
        self.split_stats = {'n_total_train_cases': [],
                            'n_included_train_cases': []}

    def _split(self, i: int) -> (pd.DataFrame, np.ndarray,
                                 pd.DataFrame, np.ndarray):
        """Train-test split, according to the pre-defined splits calculated
            in 1_train_test_split.py"""
        (X_train_df, y_train, X_test_df, y_test,
         n_total_train_cases, n_included_train_cases) = split_into_folds(
            self.df,
            indices={'train': self.tts.train_i[i], 'test': self.tts.test_i[i]},
            target_var_name=self.target_variable_name)
        self.split_stats['n_total_train_cases'].append(n_total_train_cases)
        self.split_stats['n_included_train_cases'].append(
            n_included_train_cases)
        return X_train_df, y_train, X_test_df, y_test

    def _calculate_convenience_split_stats(self):
        for stat in self.split_stats.keys():
            self.split_stats[stat] = np.array(self.split_stats[stat])
        self.split_stats['n_excluded_train_cases'] = (
            self.split_stats['n_total_train_cases'] -
            self.split_stats['n_included_train_cases'])
        self.split_stats['fraction_excluded_train_cases'] = 1 - (
            self.split_stats['n_included_train_cases'] /
            self.split_stats['n_total_train_cases'])
