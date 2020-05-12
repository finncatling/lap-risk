import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.helpers import flatten_nela_var_dict
from utils.explore import missingness_perc


class TrainTestSplitter:
    """Splits NELA data into training and testing folds. Splitting uses the
        anonymised institution (trusts or hospital) IDs such that the training
        and test folds don't contain cases from the same trust / hospital.

        Selects consistent test fold cases for with all models, by excluding
        current-NELA-model-variable incomplete cases from the test fold.

        TODO: A few end-to-end tests for sanity checking

        TODO: Consider whether current-NELA-model incomplete cases are likely
            to be systemically different from the incomplete cases.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 split_variable_name: str,
                 test_fraction: float,
                 n_splits: int,
                 current_nela_model_vars: Dict,
                 random_seed):
        """Note that the indices for the each case in self.complete_case_df
            match the corresponding cases in self.df, i.e. we don't reset the
            index of self.complete_case_df after dropping the incomplete cases.

        Args:
            df: NELA data after initial manual wrangling
            split_variable_name: Name of DataFrame column containing the
                anonymised institution IDs used in splitting
            test_fraction: Fraction of institutions from which the test fold
                cases are sourced. In interval [0, 1]
            n_iters: Number of iterations of train-test splitting to perform.
                The multiple splits obtained will be used later to calculate
                confidence intervals for the models' performance.
            current_nela_model_vars: Column names of the variables used by the
                current NELA risk model
            random_seed: Used for random selection of institution IDs
        """
        self.split_variable_name = split_variable_name
        self.test_fraction = test_fraction
        self.n_splits = n_splits
        self.rnd = np.random.RandomState(random_seed)
        self.nela_vars = flatten_nela_var_dict(current_nela_model_vars)
        self.df = self._preprocess_df(df)
        self.complete_case_df, self.drop_stats = drop_incomplete_cases(self.df)
        self._split_has_run = False
        self.train_institution_ids: List[np.ndarray] = []
        self.test_institution_ids: List[np.ndarray] = []
        self.train_i: List[np.ndarray] = []
        self.test_i: List[np.ndarray] = []
        self.split_stats = {'n_test_cases': [],
                            'test_fraction_of_complete_cases': [],
                            'test_fraction_of_total_cases': []}

    @property
    def n_institutions(self) -> int:
        return self.df[self.split_variable_name].nunique()

    @property
    def n_test_institutions(self) -> int:
        return int(np.round(self.n_institutions * self.test_fraction))

    @property
    def n_train_institutions(self) -> int:
        return self.n_institutions - self.n_test_institutions

    @property
    def institution_ids(self) -> np.ndarray:
        return self.df[self.split_variable_name].unique()

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """1) Resets index, 2) checks that split variable has no missing values
            and so doesn't change the number of complete cases by being dropped
            later, 3) removes variables not in current NELA risk model."""
        df = df.copy().reset_index(drop=True)
        assert missingness_perc(df, self.split_variable_name) == 0.0
        return df[self.nela_vars + [self.split_variable_name]]

    def split(self) -> None:
        """Each iteration represents one train-test split. Each iteration
            produces four arrays: 1) train institution IDs, 2) test institution
            IDs, 3) indices of train fold cases, 4) indices of test fold
            cases.
        """
        if self._split_has_run:
            warnings.warn('Train-test splitting has already run.')
        else:
            for _ in range(self.n_splits):
                self._split_institutions()
                self._split_cases()
                self._calculate_split_stats()
            self._split_has_run = True

    def _split_institutions(self):
        """Randomly split institution IDs into train and test subsets."""
        self.test_institution_ids.append(np.sort(self.rnd.choice(
            self.institution_ids, self.n_test_institutions, replace=False)))
        self.train_institution_ids.append(np.array(list(
            set(self.institution_ids) - set(self.test_institution_ids[-1]))))

    def _split_cases(self):
        """Finds indices of train fold cases (includes incomplete cases, which
            can be removed later using the split_into_folds function) and
            indices of test fold cases (excludes current-NELA-model-variable
            incomplete cases)."""
        self.train_i.append(self.df.index[self.df[
            self.split_variable_name].isin(
            self.train_institution_ids[-1])].to_numpy())
        self.test_i.append(self.complete_case_df.index[self.complete_case_df[
            self.split_variable_name].isin(
            self.test_institution_ids[-1])].to_numpy())

    def _calculate_split_stats(self):
        self.split_stats['n_test_cases'].append(self.test_i[-1].shape[0])
        self.split_stats['test_fraction_of_complete_cases'].append(
            self.test_i[-1].shape[0] / self.drop_stats['n_complete_cases'])
        self.split_stats['test_fraction_of_total_cases'].append(
            self.test_i[-1].shape[0] / self.drop_stats['n_total_cases'])


def drop_incomplete_cases(df: pd.DataFrame) -> (pd.DataFrame, Dict[str, float]):
    """Drops incomplete rows in input DataFrame, keeping track of pre- and
        post-drop case numbers and calculating some convenience stats."""
    drop_stats = {'n_total_cases': df.shape[0]}
    df = df.dropna()  # DON'T reset index
    drop_stats['n_complete_cases'] = df.shape[0]
    drop_stats['n_dropped_cases'] = (drop_stats['n_total_cases'] -
                                     drop_stats['n_complete_cases'])
    drop_stats['fraction_dropped'] = (1 - drop_stats['n_complete_cases'] /
                                      drop_stats['n_total_cases'])
    return df, drop_stats


def split_into_folds(
        df: pd.DataFrame,
        indices: Dict[str, np.ndarray],
        target_var_name: str
) -> (pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, int, int):
    """Splits supplied DataFrame into train and test folds, such that the test
        fold is the cases from the test trusts which are complete for the
        variables in the current NELA risk model, and the train fold is all
        the available cases from the train trusts. Two things to note:

        1) The train fold will be different between models as for the current
        NELA model it will only contain complete cases, whereas for our novel
        model it will also contain the cases that were incomplete prior to
        imputation.

        2) The test fold will be the same for all models, i.e. the current-NELA-
        model incomplete cases from the test fold trusts will not be used in
        training or testing of any of the models.

    Args:
        df: NELA data
        indices: Of the form {'train': [train_fold_indices...],
                              'test': [test_fold_indices...]}
        target_var_name: Name of DataFrame column containing mortality status

    Returns:
        Training features
        Training targets
        Testing features
        Testing targets
        Number of cases in unabridged train fold
        Number of train-fold cases available in the input DataFrame, i.e. the
            number of cases in the returned training data.
    """
    n_total_train_cases = indices['train'].shape[0]
    indices['train'] = np.array([i for i in indices['train'] if i in df.index])
    n_intersection_train_cases = indices['train'].shape[0]

    split = {}
    for fold in ('train', 'test'):
        split[fold] = {'X_df': df.loc[
            indices[fold]].copy().reset_index(drop=True)}
        split[fold]['y'] = split[fold]['X_df'][target_var_name].values
        split[fold]['X_df'] = split[fold]['X_df'].drop(target_var_name, axis=1)

    assert split['test']['X_df'].shape[0] == indices['test_i'].shape[0]

    return (split['train']['X_df'], split['train']['y'],
            split['test']['X_df'], split['test']['y'],
            n_total_train_cases, n_intersection_train_cases)
