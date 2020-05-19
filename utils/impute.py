import copy
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
from numpy.random import RandomState
from progressbar import progressbar as pb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from statsmodels.imputation.mice import MICEData
from statsmodels.discrete import discrete_model

from utils.split import Splitter, TrainTestSplitter
from utils.model.novel import winsorize_novel


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
        if len(self.n_imputations):
            self.n_imputations.append(
                int(np.ceil(n_min_imputations / self.n_imputations[-1]) *
                    self.n_imputations[-1]))
        else:
            self.n_imputations.append(n_min_imputations)


class SplitterWinsorMICE(Splitter):
    """Performs winsorization then MICE for each predefined train-test split.
        MICE is limited to the variables identified in the ImputationInfo for
        the first imputation stage. For efficiency, we store only the imputed
        values and their indices.

        NB. statsmodels doesn't provide the option to pass a random seed,
        so the MICE outputs will differ each of each run"""
    def __init__(self,
                 df: pd.DataFrame,
                 test_train_splitter: TrainTestSplitter,
                 target_variable_name: str,
                 winsor_variables: List[str],
                 winsor_quantiles: Tuple[float, float],
                 winsor_include: Dict[str, Tuple[bool, bool]],
                 n_imputations: int,
                 binary_variables: List[str],
                 n_burn_in: int,
                 n_skip: int):
        super().__init__(df, test_train_splitter, target_variable_name)
        self.winsor_variables = winsor_variables
        self.winsor_quantiles = winsor_quantiles
        self.winsor_include = winsor_include
        self.n_imputations = n_imputations
        self.binary_vars = binary_variables
        self.n_burn_in = n_burn_in
        self.n_skip = n_skip
        self.winsor_thresholds: List[Dict[str, Tuple[float, float]]] = []
        self.missing_i: Dict[str, List[Dict[str, np.ndarray]]] = {
            'train': [], 'test': []}
        self.imputed: Dict[str, List[List[Dict[str, np.ndarray]]]] = {
            'train': [], 'test': []}

    def run_mice(self):
        """Runs MICE for train and test folds of all train-test splits."""
        # for i in pb(range(self.tts.n_splits), prefix='Split iteration'):
        # TODO: Change this line back (changed to shorter loop for testing)
        for i in pb(range(2), prefix='Split iteration'):
            X_train_df, _, X_test_df, _ = self._split(i)
            X_train_df, X_test_df = self._winsorize(X_train_df, X_test_df)
            X_dfs = {'train': X_train_df, 'test': X_test_df}
            for fold in ('train', 'test'):
                self._single_df_mice(X_dfs[fold], fold)

    # TODO: Method to reconstruct imputed dataframes

    def _winsorize(
        self, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        X_train_df, winsor_thresholds = winsorize_novel(
            X_train_df,
            cont_vars=self.winsor_variables,
            quantiles=self.winsor_quantiles,
            include=self.winsor_include)
        X_test_df, _ = winsorize_novel(X_test_df,
                                       thresholds=winsor_thresholds,
                                       cont_vars=self.winsor_variables)
        self.winsor_thresholds.append(winsor_thresholds)
        return X_train_df, X_test_df

    def _single_df_mice(self, X_df: pd.DataFrame, fold: str):
        """Run MICE for a single fold from a single train-test split."""
        mice_data = MICEData(X_df)
        self.missing_i[fold].append(copy.deepcopy(mice_data.ix_miss))
        for v in self.binary_vars:
            mice_data.set_imputer(v, model_class=discrete_model.Logit)

        # Initial MICE 'burn in' imputations which we discard
        for _ in range(self.n_burn_in):
            mice_data.update_all()

        self.imputed[fold].append([])
        for i in range(self.n_imputations):
            if i:
                # skip some MICE imputations between the ones we keep
                mice_data.update_all(self.n_skip + 1)
            self._store_imputed(mice_data.data, fold)

    def _store_imputed(self, imp_df: pd.DataFrame, fold: str):
        """Store just the imputed values from a single MICE iteration."""
        self.imputed[fold][-1].append({})
        for col_name, missing_i in self.missing_i[fold][-1].items():
            self.imputed[fold][-1][-1] = imp_df.iloc[
                missing_i, imp_df.columns.get_loc(col_name)].values


class CategoricalImputer:
    """Imputes missing values of non-binary categorical
        variables in MICE DataFrames."""

    def __init__(self,
                 original_df: pd.DataFrame,
                 mice_dfs: List[pd.DataFrame],
                 cont_vars: List[str],
                 binary_vars: List[str],
                 cat_vars: List[str],
                 random_seed):
        """Args:
            original_df: DataFrame containing all variables
                prior to imputation i.e. with all missing
                values still present. 
            mice_dfs: Output of (previously run) statsmodels MICE
                for continuous and binary variables, i.e. each DataFrame
                has columns for each variable in cont_vars and
                binary_vars only
            cont_vars: Continuous variables
            binary_vars: Binary variables
            cat_vars: Non-binary categorical variables
            random_seed: Should produce consistent results
                between runs, providing input arguments are
                the same
        """
        self.odf = original_df.copy()
        self.mice_dfs = copy.deepcopy(mice_dfs)
        self.cont_vars = cont_vars
        self.binary_vars = binary_vars
        self.cat_vars = cat_vars
        self.random_seed = random_seed
        self._v = {}
        self.imputed_dfs = copy.deepcopy(mice_dfs)
        self._rnd = self._init_rnd()

    @property
    def n_mice_dfs(self) -> int:
        return len(self.mice_dfs)

    def impute_all(self):
        self._preprocess_dfs()
        for v in self.cat_vars:
            self._impute_v(v)

    def _preprocess_dfs(self):
        self._reset_df_indices()
        self._reorder_df_columns()
        self._scale_mice_dfs()

    def _reset_df_indices(self):
        self.odf = self.odf.reset_index(drop=True)
        for i in range(len(self.mice_dfs)):
            self.mice_dfs[i] = self.mice_dfs[i].reset_index(drop=True)

    def _reorder_df_columns(self):
        """Reordering isn't strictly necessary here, but allows us to
            check that we don't have additional / missing columns in
            the input DataFrames."""
        n_cols = len(self.odf.columns)
        self.odf = self.odf[self.cont_vars + self.binary_vars +
                            self.cat_vars]
        assert (n_cols == self.odf.shape[1])
        for i in range(len(self.mice_dfs)):
            n_cols = len(self.mice_dfs[i].columns)
            self.mice_dfs[i] = self.mice_dfs[i][self.cont_vars +
                                                self.binary_vars]
            assert (n_cols == self.mice_dfs[i].shape[1])

    def _scale_mice_dfs(self):
        """We need to scale the continuous variables in order to use
            fast solvers for multinomial logistic regression in sklearn"""
        for i in range(self.n_mice_dfs):
            s = RobustScaler()
            self.mice_dfs[i].loc[:, self.cont_vars] = s.fit_transform(
                self.mice_dfs[i].loc[:, self.cont_vars].values)

    def _impute_v(self, v: str):
        self._v[v] = {'imp': []}
        self._get_train_missing_i(v)
        self._get_y_train(v)
        if self._v[v]['missing_i'].shape[0]:
            for i in pb(range(self.n_mice_dfs), prefix=f'{v}'):
                y_missing_probs, y_classes = self._pred_y_missing_probs(v, i)
                self._impute_y(v, i, y_missing_probs, y_classes)
        else:
            print(f'Skipping {v} imputation as no missing values')
        self._update_imputed_dfs(v)

    def _get_train_missing_i(self, v: str):
        self._v[v]['train_i'] = self.odf.loc[self.odf[v].notnull()].index
        self._v[v]['missing_i'] = self.odf.loc[self.odf[v].isnull()].index

    def _get_y_train(self, v: str):
        self._v[v]['y_train'] = self.odf.loc[self._v[v]['train_i'], v].values

    def _pred_y_missing_probs(self, v: str, i: int) -> (np.ndarray, np.ndarray):
        """1st return is of shape (n_samples, n_classes), where each row
            corresponds to a missing value of v, and each columns is
            the predicted probability that the missing value is that
            class."""
        lr = LogisticRegression(penalty='none', solver='sag', max_iter=3000,
                                multi_class='multinomial', n_jobs=16,
                                random_state=self._rnd['lr'])
        lr.fit(self.mice_dfs[i].loc[self._v[v]['train_i']].values,
               self._v[v]['y_train'])
        return (lr.predict_proba(self.mice_dfs[i].loc[
                                     self._v[v]['missing_i']].values),
                lr.classes_)

    def _impute_y(self, v: str, i: int, y_missing_probs, y_classes):
        """Rather than imputing each missing value using
            idxmax(y_missing_probs), we impute each missing value
            probabilistically using the predicted probabilities."""
        self._v[v]['imp'].append(np.zeros(self._v[v]['missing_i'].shape[0]))
        for j in range(self._v[v]['imp'][i].shape[0]):
            self._v[v]['imp'][i][j] = self._rnd['choice'].choice(
                y_classes, p=y_missing_probs[j, :])

    def _update_imputed_dfs(self, v):
        for i in range(self.n_mice_dfs):
            self.imputed_dfs[i][v] = np.zeros(self.imputed_dfs[i].shape[0])
            self.imputed_dfs[i].loc[
                self._v[v]['train_i'], v] = self._v[v]['y_train']
            if self._v[v]['missing_i'].shape[0]:
                self.imputed_dfs[i].loc[
                    self._v[v]['missing_i'], v] = self._v[v]['imp'][i]

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
