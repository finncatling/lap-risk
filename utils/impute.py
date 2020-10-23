from typing import List, Dict

import numpy as np
import pandas as pd

from utils.split import Splitter, TrainTestSplitter


def determine_n_imputations(df: pd.DataFrame) -> (int, float):
    """White et al (https://pubmed.ncbi.nlm.nih.gov/21225900/), Section 7.3
        recommends the following rule of thumb: The number of MICE imputations
        should be at least 100 * f MICE imputations, where f is the fraction of
        incomplete cases in the DataFrame."""
    fraction_incomplete = 1 - (df.dropna(how="any").shape[0] / df.shape[0])
    n_imputations = int(np.ceil(fraction_incomplete * 100))
    return n_imputations, fraction_incomplete


def find_missing_indices(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Finds indices of missing values for each variable in the DataFrame.
        Adapted from implementation in statsmodels MICEData"""
    missing_i = {}
    for col in df.columns:
        null = pd.isnull(df[col])
        missing_i[col] = np.flatnonzero(null)
    return missing_i


class ImputationInfo:
    """Hold info related to a (possibly multi-stage) imputation process.	
        The number of imputations calculated for a subsequent stage is always	
        a multiple of the number needed in the previous stage."""

    def __init__(self):
        self.descriptions: List[str] = []
        self.n_min_imputations: List[int] = []
        self.fraction_incomplete: List[float] = []
        self.n_imputations: List[int] = []
        self.multiple_of_previous_n_imputations: List[int] = []

    def add_stage(self, description: str, df: pd.DataFrame) -> None:
        """Add information about an imputation stage, and calculate the
            number of imputations it will require.

            Args:
                description: Description of this imputation stage
                df: Containing variables to be imputed during this stage. May
                    also contain other variables as long as they have no
                    missing values
        """
        self.descriptions.append(description)
        self._determine_adjusted_n_imputations(df)

    def _determine_adjusted_n_imputations(self, df: pd.DataFrame):
        """If there is a previous imputation stage, increase n_imputations
            (the number of imputations required for this stage according to
            White et al) so that it is a multiple of n_imputations from the
            previous stage."""
        n_min_imputations, fraction_incomplete = determine_n_imputations(df)
        self.n_min_imputations.append(n_min_imputations)
        self.fraction_incomplete.append(fraction_incomplete)
        multiple = 1
        if len(self.n_imputations):
            multiple = int(np.ceil(n_min_imputations / self.n_imputations[-1]))
            self.n_imputations.append(multiple * self.n_imputations[-1])
        else:
            self.n_imputations.append(n_min_imputations)
        self.multiple_of_previous_n_imputations.append(multiple)


class Imputer(Splitter):
    """Base class for imputers."""
    def __init__(
        self,
        df: pd.DataFrame,
        train_test_splitter: TrainTestSplitter,
        target_variable_name: str
    ):
        """
        Args:
            df: Preprocessed NELA data
            train_test_splitter: TrainTestSplitter instance from
                01_train_test_split.py
            target_variable_name: Name of DataFrame column containing mortality
                status
        """
        super().__init__(df, train_test_splitter, target_variable_name)
        self.missing_i: Dict[str,  # fold name
                             Dict[int,  # train-test split index
                                  Dict[str,  # variable name
                                       np.ndarray]]] = {'train': {},
                                                        'test': {}}
        self.imputed: Dict[str,  # fold name
                           Dict[int,  # train-test split index
                                Dict[int,  # MICE imputation index
                                     Dict[str,  # variable name
                                          np.ndarray]]]] = {'train': {},
                                                            'test': {}}

    def _find_missing_indices(
        self,
        split_i: int,
        train: pd.DataFrame,
        test: pd.DataFrame,
        variable_names: List[str]
    ):
        """Find the indices (for the indexed train-test split) of the missing
            values of the specified variables. These missing values are
            consistent across MICE imputations."""
        for fold_name, df in {"train": train, "test": test}.items():
            self.missing_i[fold_name][split_i] = find_missing_indices(
                df[variable_names]
            )

    def _split_then_join_Xy(self, i: int) -> (pd.DataFrame, pd.DataFrame):
        """Thin wrapper around ._split() which adds y_train (the mortality
            labels) back into X_train, and adds y_test back into X_test.
            This is convenient for input to the MICE and categorical
            imputation models, which use the target as a feature."""
        train, y_train, test, y_test = self._split(i)
        train[self.target_variable_name] = y_train
        test[self.target_variable_name] = y_test
        return train, test

    def get_imputed_variables(
        self, fold_name: str, split_i: int, imp_i: int
    ) -> pd.DataFrame:
        """Reconstructs imputed DataFrame containing all the POTENTIALLY
            imputed variables from a given imputation iteration, for a given
            fold, from a given train-test split.

            In MICE, the potentially imputed variables are all the variables in
            the input DataFrame (including the target). In CategoricalImputer,
            they are just the non-binary categorical variables.

        Args:
            split_i: Train-test split iteration index
            fold_name: in ('train', 'test')
            imp_i: Imputation (e.g. MICE iteration) index for this train-test
                split iteration

        Returns:
            DataFrame with imputed values
        """
        if fold_name == "train":
            imp_df, _ = self._split_then_join_Xy(split_i)
        elif fold_name == "test":
            _, imp_df = self._split_then_join_Xy(split_i)
        imp_df = imp_df[list(self.missing_i[fold_name][split_i].keys())]
        for var_name, imp in self.imputed[fold_name][split_i][imp_i].items():
            imp_df.iloc[
                self.missing_i[fold_name][split_i][var_name],
                imp_df.columns.get_loc(var_name)
            ] = imp
        assert imp_df.dropna(how='any').shape[0] == imp_df.shape[0]
        return imp_df
