import copy
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
from numpy.random import RandomState
from progressbar import progressbar as pb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from statsmodels.imputation.mice import MICEData
from statsmodels.regression import linear_model
from statsmodels.discrete import discrete_model

from utils.split import Splitter, TrainTestSplitter
from utils.model.novel import winsorize_folds_novel


def determine_n_imputations(df: pd.DataFrame) -> (int, float):
    """White et al recommend using 100 * f MICE imputations, where f is the
        fraction of incomplete cases in the DataFrame."""
    fraction_incomplete = 1 - (df.dropna(how='any').shape[0] / df.shape[0])
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
    def __init__(self, overall_description: Union[None, str] = None):
        self.overall_description = overall_description
        self.descriptions: List[str] = []
        self.impute_vars: List[List[str]] = []
        self.all_vars: List[List[str]] = []
        self.n_min_imputations: List[int] = []
        self.fraction_incomplete: List[float] = []
        self.n_imputations: List[int] = []
        self.multiple_of_previous_n_imputations: List[int] = []

    def add_stage(self,
                  description: str,
                  df: pd.DataFrame,
                  variables_to_impute: List[str]) -> None:
        """Add information about an imputation stage, and calculate the number
            of imputations it will require.

        Args:
            description: Description of this imputation stage
            df: Data used in this imputation stage, i.e. variables to be
                imputed and variables used as features in the imputation
                model (in MICE these two variable sets intersect). May also
                contain complete variables which are unused as imputation
                features
            variables_to_impute: Names of variables to be imputed in this
                stage
        """
        all_vars = list(df.columns)
        self._sanity_check(variables_to_impute, all_vars)
        self.descriptions.append(description)
        self.impute_vars.append(variables_to_impute)
        self.all_vars.append(all_vars)
        self._determine_adjusted_n_imputations(df)

    @staticmethod
    def _sanity_check(impute_vars: List[str], all_vars: List[str]):
        for var in impute_vars:
            assert var in all_vars

    def _determine_adjusted_n_imputations(self, df: pd.DataFrame):
        """If there is a previous imputation stage, increase n_imputations (the
            number of imputations required for this stage according to White et
            al) so that it is a multiple of n_imputations from the previous
            stage."""
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


class SplitterWinsorMICE(Splitter):
    """Performs winsorization then MICE for each predefined train-test split.
        MICE is limited to the variables identified in the ImputationInfo for
        the first imputation stage. For efficiency, we store only the imputed
        values and their indices.

        NB. statsmodels doesn't provide the option to pass a random seed,
        so the MICE outputs will differ each of each run"""
    def __init__(self,
                 df: pd.DataFrame,
                 train_test_splitter: TrainTestSplitter,
                 target_variable_name: str,
                 cont_variables: List[str],
                 binary_variables: List[str],
                 winsor_quantiles: Tuple[float, float],
                 winsor_include: Dict[str, Tuple[bool, bool]],
                 n_mice_imputations: int,
                 n_mice_burn_in: int,
                 n_mice_skip: int):
        super().__init__(df, train_test_splitter, target_variable_name)
        self.cont_vars = cont_variables
        self.binary_vars = binary_variables
        self._sanity_check_variables()
        self.winsor_quantiles = winsor_quantiles
        self.winsor_include = winsor_include
        self.n_mice_imputations = n_mice_imputations
        self.n_mice_burn_in = n_mice_burn_in
        self.n_mice_skip = n_mice_skip
        self.winsor_thresholds: Dict[int,  # train-test split index
                                     Dict[str, Tuple[float, float]]] = {}
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

    @property
    def all_vars(self):
        return self.cont_vars + self.binary_vars

    def _sanity_check_variables(self):
        """Check that, apart from target (accounted for with the minus 1), df
            only contains the continuous and binary variables specified."""
        assert len(self.all_vars) == len(self.df.columns) - 1
        for var in self.all_vars:
            assert var in self.df.columns

    def split_winsorize_mice(self):
        """Split df according to pre-defined train-test splits, perform
            winsorization (using thresolds from train fold to winsorize test
            fold), then run MICE for train and test folds of all train-test
            splits."""
        for i in pb(range(self.tts.n_splits), prefix='Split iteration'):
            X_train_df, _, X_test_df, _ = self._split(i)
            X_train_df, X_test_df, winsor_thresholds = winsorize_folds_novel(
                X_train_df,
                X_test_df,
                cont_vars=self.cont_vars,
                quantiles=self.winsor_quantiles,
                include=self.winsor_include)
            self.winsor_thresholds[i] = winsor_thresholds
            X_dfs = {'train': X_train_df, 'test': X_test_df}
            for fold in ('train', 'test'):
                self._single_fold_mice(i, fold, X_dfs[fold])

    def get_imputed_df(self, split_i: int, fold: str, imputation_i: int):
        """Construct imputed DataFrame from a given imputation iteration, for a
            given fold, from a given train-test split."""
        if fold == 'train':
            imp_df, _, _, _ = self._split(split_i)
        elif fold == 'test':
            _, _, imp_df, _ = self._split(split_i)
        else:
            raise Exception("fold not in ('train', 'test')")
        for var_name, missing_i in self.missing_i[fold][split_i].items():
            imp_df.iloc[missing_i, imp_df.columns.get_loc(var_name)] = (
                self.imputed[fold][split_i][imputation_i][var_name])
        return imp_df

    def _single_fold_mice(self, split_i: int, fold: str, X_df: pd.DataFrame):
        """Set up and run MICE for a single fold from a single train-test
            split."""
        mice_data = MICEData(X_df)
        self.missing_i[fold][split_i] = copy.deepcopy(mice_data.ix_miss)
        mice_data = self._set_mice_imputers(mice_data)
        self.imputed[fold][split_i] = {}  # dict will hold fold's imputed values
        self._run_mice_loop(split_i, fold, mice_data)

    def _set_mice_imputers(self, mice_data: MICEData) -> MICEData:
        for var in self.cont_vars:
            mice_data.set_imputer(var, model_class=linear_model.OLS,
                                  fit_kwds={'disp': False})
        for var in self.binary_vars:
            mice_data.set_imputer(var, model_class=discrete_model.Logit,
                                  fit_kwds={'disp': False})
        return mice_data

    def _run_mice_loop(self, split_i: int, fold: str, mice_data: MICEData):
        """'Burn-in' and 'skip' imputations are discarded."""
        for _ in range(self.n_mice_burn_in):
            mice_data.update_all()
        for imputation_i in range(self.n_mice_imputations):
            if imputation_i:
                mice_data.update_all(self.n_mice_skip + 1)
            self._store_imputed(split_i, fold, imputation_i, mice_data.data)

    def _store_imputed(self, split_i: int, fold: str, imputation_i: int,
                       imp_df: pd.DataFrame):
        """Store just the imputed values from a single MICE iteration."""
        self.imputed[fold][split_i][imputation_i] = {}
        for var_name, missing_i in self.missing_i[fold][split_i].items():
            self.imputed[fold][split_i][imputation_i][var_name] = imp_df.iloc[
                missing_i, imp_df.columns.get_loc(var_name)].copy().values


class CategoricalImputer(Splitter):
    """Imputes missing values of non-binary categorical variables, using
        output of earlier MICE."""

    def __init__(self,
                 df: pd.DataFrame,
                 train_test_splitter: TrainTestSplitter,
                 target_variable_name: str,
                 splitter_winsor_mice: SplitterWinsorMICE,
                 cont_vars: List[str],
                 binary_vars: List[str],
                 cat_vars: List[str],
                 n_imputations_per_mice: int,
                 random_seed):
        """Args:
            df: DataFrame containing all continuous variables (except lactate-
                and albumin-related variables), all binary variables, the
                non-binary discrete variables for imputation at this stage, and
                the target (mortality labels). This DataFrame still contains all
                its missing values, i.e. no imputation yet
            splitter_winsor_mice: Pickled SplitterWinsorMice object containing
                the results of MICE for the continuous variables (except lactate
                and albumin) and the binary variables
            cont_vars: Continuous variables (excluding lactate and albumin)
            binary_vars: Binary variables (excluding lactate and albumin
                missingness indicators)
            cat_vars: Non-binary categorical variables for imputation
            random_seed: For reproducibility
        """
        super().__init__(df, train_test_splitter, target_variable_name)
        self.swm = splitter_winsor_mice
        self.cont_vars = cont_vars
        self.binary_vars = binary_vars
        self.cat_vars = cat_vars
        self.imp_multiple = n_imputations_per_mice
        self.random_seed = random_seed
        self._v = {}  # TODO: rename and add typing
        self._rnd = self._init_rnd()
        self.missing_i: Dict[str,  # fold name
                             Dict[int,  # train-test split index
                                  Dict[str,  # variable name
                                       np.ndarray]]] = {'train': {},
                                                        'test': {}}
        # TODO: get_imputed_df method

    def fit_all_imputers(self):
        for i in pb(range(self.tts.n_splits), prefix='Split iteration'):
            X_train_df, _, X_test_df, _ = self._split(i)
            # TODO: winsorise
            # TODO: finish method



    # @property
    # def n_mice_dfs(self) -> int:
    #     return len(self.mice_dfs)
    #
    # def impute_all(self):
    #     self._preprocess_dfs()
    #     for v in self.cat_vars:
    #         self._impute_v(v)
    #
    # def _preprocess_dfs(self):
    #     self._reset_df_indices()
    #     self._reorder_df_columns()
    #     self._scale_mice_dfs()
    #
    # def _reset_df_indices(self):
    #     self.df = self.df.reset_index(drop=True)
    #     for i in range(len(self.mice_dfs)):
    #         self.mice_dfs[i] = self.mice_dfs[i].reset_index(drop=True)
    #
    # def _scale_mice_dfs(self):
    #     """We need to scale the continuous variables in order to use
    #         fast solvers for multinomial logistic regression in sklearn"""
    #     for i in range(self.n_mice_dfs):
    #         s = RobustScaler()
    #         self.mice_dfs[i].loc[:, self.cont_vars] = s.fit_transform(
    #             self.mice_dfs[i].loc[:, self.cont_vars].values)
    #
    # def _impute_v(self, v: str):
    #     self._v[v] = {'imp': []}
    #     self._get_train_missing_i(v)
    #     self._get_y_train(v)
    #     if self._v[v]['missing_i'].shape[0]:
    #         for i in pb(range(self.n_mice_dfs), prefix=f'{v}'):
    #             y_missing_probs, y_classes = self._pred_y_missing_probs(v, i)
    #             self._impute_y(v, i, y_missing_probs, y_classes)
    #     else:
    #         print(f'Skipping {v} imputation as no missing values')
    #     self._update_imputed_dfs(v)
    #
    # def _get_train_missing_i(self, v: str):
    #     self._v[v]['train_i'] = self.df.loc[self.df[v].notnull()].index
    #     self._v[v]['missing_i'] = self.df.loc[self.df[v].isnull()].index
    #
    # def _get_y_train(self, v: str):
    #     self._v[v]['y_train'] = self.df.loc[self._v[v]['train_i'], v].values
    #
    # def _pred_y_missing_probs(self, v: str, i: int) -> (np.ndarray, np.ndarray):
    #     """1st return is of shape (n_samples, n_classes), where each row
    #         corresponds to a missing value of v, and each columns is
    #         the predicted probability that the missing value is that
    #         class."""
    #     lr = LogisticRegression(penalty='none', solver='sag', max_iter=3000,
    #                             multi_class='multinomial', n_jobs=16,
    #                             random_state=self._rnd['lr'])
    #     lr.fit(self.mice_dfs[i].loc[self._v[v]['train_i']].values,
    #            self._v[v]['y_train'])
    #     return (lr.predict_proba(self.mice_dfs[i].loc[
    #                                  self._v[v]['missing_i']].values),
    #             lr.classes_)
    #
    # def _impute_y(self, v: str, i: int, y_missing_probs, y_classes):
    #     """Rather than imputing each missing value using
    #         idxmax(y_missing_probs), we impute each missing value
    #         probabilistically using the predicted probabilities."""
    #     self._v[v]['imp'].append(np.zeros(self._v[v]['missing_i'].shape[0]))
    #     for j in range(self._v[v]['imp'][i].shape[0]):
    #         self._v[v]['imp'][i][j] = self._rnd['choice'].choice(
    #             y_classes, p=y_missing_probs[j, :])
    #
    # def _update_imputed_dfs(self, v):
    #     for i in range(self.n_mice_dfs):
    #         self.imputed_dfs[i][v] = np.zeros(self.imputed_dfs[i].shape[0])
    #         self.imputed_dfs[i].loc[
    #             self._v[v]['train_i'], v] = self._v[v]['y_train']
    #         if self._v[v]['missing_i'].shape[0]:
    #             self.imputed_dfs[i].loc[
    #                 self._v[v]['missing_i'], v] = self._v[v]['imp'][i]

    def _init_rnd(self) -> Dict[str, RandomState]:
        return {'lr': RandomState(self.random_seed),
                'choice': RandomState(self.random_seed)}


class ContImpPreprocessor:
    """Prepares output from notebook 4 (list of DataFrames from MICE and
        categorical imputation) for input to GAMs for imputation of
        remaining unimputed variables, e.g. lactate, albumin."""

    def __init__(self,
                 imp_dfs: List[pd.DataFrame],
                 target: str,
                 drop_vars: List[str]):
        """Args: 
            imp_dfs: Output of imputation in notebook 4. Each DataFrame
                contains complete columns for all variables except a select
                few that have missing values and will be imputed in production
                if missing, e.g. lactate, albumin.                
            target: The variable from imp_dfs for imputation. The indices of
                the missing values for target are assumed to be consistent
                across imp_dfs
            drop_vars: Variables from imp_dfs not to include in the imputation
                models
        """
        self.imp_dfs = copy.deepcopy(imp_dfs)
        self.target = target
        self.drop_vars = drop_vars
        self._i = None
        self.X = {'train': [], 'missing': []}
        self.y_train = None
        self.y_train_trans = None

    @property
    def n_imp_dfs(self) -> int:
        return len(self.imp_dfs)

    def preprocess(self):
        self._i = self._get_train_missing_i()
        self._y_train_split()
        for i in pb(range(self.n_imp_dfs)):
            self._drop_unused_variables(i)
            self._X_train_missing_split(i)

    def yield_train_X_y(self, i, trans_y=False):
        """Optionally yields a transformed version of y (this
            transformation currently has to occur by some
            method external to this class."""
        if trans_y:
            return self.X['train'][i].values, self.y_train_trans
        else:
            return self.X['train'][i].values, self.y_train

    def _get_train_missing_i(self) -> Dict[str, pd.Int64Index]:
        return {'train': self.imp_dfs[0].loc[self.imp_dfs[0][
            self.target].notnull()].index,
                'missing': self.imp_dfs[0].loc[self.imp_dfs[0][
                    self.target].isnull()].index}

    def _drop_unused_variables(self, i: int):
        self.imp_dfs[i] = self.imp_dfs[i].drop(self.drop_vars, axis=1)

    def _y_train_split(self):
        """y hasn't been imputed yet, so should be the same in all
            imp_dfs."""
        self.y_train = copy.deepcopy(
            self.imp_dfs[0].loc[self._i['train'], self.target].values)
        for i in range(1, self.n_imp_dfs):  # Check that all y are same
            assert ((self.y_train ==
                     self.imp_dfs[i].loc[self._i['train'],
                                         self.target].values).all())

    def _X_train_missing_split(self, i: int):
        for fold in self.X.keys():
            self.X[fold].append(self.imp_dfs[i].loc[
                                    self._i[fold]].drop(self.target,
                                                        axis=1).copy())
