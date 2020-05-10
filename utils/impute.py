# -*- coding: utf-8 -*-
import copy
from typing import List, Dict

import numpy as np
import pandas as pd
from numpy.random import RandomState
from progressbar import progressbar as pb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


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
            fast solvers for multinomial logistic regression in sklearn:"""
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
        """ 1st return is of shape (n_samples, n_classes), where each row
            corresponds to a missing value of v, and each columns is
            the predicted probabity that the missing value is that
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

    @property
    def n_mice_dfs(self) -> int:
        return len(self.mice_dfs)

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

    @property
    def n_imp_dfs(self) -> int:
        return len(self.imp_dfs)
