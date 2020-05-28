import copy
from typing import List, Dict, Union, Tuple, Any, Callable, Type

import numpy as np
import pandas as pd
from progressbar import progressbar as pb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from statsmodels.imputation.mice import MICEData
from statsmodels.regression import linear_model
from statsmodels.discrete import discrete_model
from pygam import GAM
from sklearn.preprocessing import QuantileTransformer

from utils.split import Splitter, TrainTestSplitter
from utils.model.novel import winsorize_novel
from utils.model.albumin import GammaTransformer
from utils.gam import combine_mi_gams


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
        """
        Args:
            df: DataFrame containing all continuous variables (except lactate-
                and albumin-related variables), all binary variables, the
                non-binary discrete variables for imputation at this stage, and
                the target (mortality labels). This DataFrame still contains all
                its missing values, i.e. no imputation yet
            train_test_splitter: Pickled TrainTestSplitter object fit in earlier
                parts of the analysis
            target_variable_name: Name of the mortality variable
            cont_variables: Continuous variables (excluding lactate and albumin)
            binary_variables: Binary variables (excluding lactate and albumin
                missingness indicators)
            winsor_quantiles: Lower and upper quantiles to winsorize
                continuous variables at by default
            winsor_include: Optionally disables winsorization at lower and/or
                upper quantiles for specified variables
            n_mice_imputations: Number of MICE-imputed DataFrames that will be
                retained for later use after running .split_winsor_mice()
            n_mice_burn_in: Number of MICE-imputed DataFrames that will be
                discarded before the first retained DataFrame
            n_mice_skip: Number of MICE-imputed DataFrames that will be
                discarded betweem retained DataFrames
        """
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
            X_train, _, X_test, _ = self._split(i)
            X_train, X_test = self._winsorize(i, X_train, X_test)
            X_dfs = {'train': X_train, 'test': X_test}
            for fold in ('train', 'test'):
                self._single_fold_mice(i, fold, X_dfs[fold])

    def get_imputed_X_df(
            self, split_i: int, fold: str, mice_imp_i: int
    ) -> pd.DataFrame:
        """Reconstructs imputed DataFrame from a given MICE iteration, for a
            given fold, from a given train-test split.

        Args:
            split_i: Train-test split iteration index
            fold: in ('train', 'test')
            mice_imp_i: MICE imputation index (for this train-test split)

        Returns:
            Columns are self.cont_vars + self.binary_vars
        """
        if fold == 'train':
            imp_df, _, _, _ = self._split(split_i)
        elif fold == 'test':
            _, _, imp_df, _ = self._split(split_i)
        for var_name, missing_i in self.missing_i[fold][split_i].items():
            imp_df.iloc[missing_i, imp_df.columns.get_loc(var_name)] = (
                self.imputed[fold][split_i][mice_imp_i][var_name])
        return imp_df

    def _winsorize(
        self, split_i: int, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        X_train, winsor_thresholds = winsorize_novel(
            X_train,
            cont_vars=self.cont_vars,
            quantiles=self.winsor_quantiles,
            include=self.winsor_include)
        X_test, _ = winsorize_novel(X_test, thresholds=winsor_thresholds)
        self.winsor_thresholds[split_i] = winsor_thresholds
        return X_train, X_test

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
                 splitter_winsor_mice: SplitterWinsorMICE,
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
            cat_vars: Non-binary categorical variables for imputation
            n_imputations_per_mice: Number of imputed datasets that will need to
                be produced by this class, expressed as a multiple of the
                number of MICE imputations already carried out
            random_seed: For reproducibility
        """
        super().__init__(df, splitter_winsor_mice.tts,
                         splitter_winsor_mice.target_variable_name)
        self.swm = splitter_winsor_mice
        self.cat_vars = cat_vars
        self.imp_multiple = n_imputations_per_mice
        self.random_seed = random_seed
        self.missing_i: Dict[str,  # fold name
                             Dict[int,  # train-test split index
                                  Dict[str,  # variable name
                                       np.ndarray]]] = {'train': {}, 'test': {}}
        self._scalers: Dict[int,  # train-test split index
                            Dict[int,  # MICE imputation index
                                 RobustScaler]] = {}
        self._imputers: Dict[int,  # train-test split index
                             Dict[int,  # MICE imputation index
                                  Dict[str,  # variable name
                                       LogisticRegression]]] = {}

    def fit(self):
        """Fit imputers for every non-binary categorical variable, in every MICE
            imputation, in every train-test split."""
        for i in pb(range(self.tts.n_splits), prefix='Split iteration'):
            self._single_train_test_split(i)

    def _single_train_test_split(self, split_i: int):
        """Fit imputers for every non-binary categorical variable, in every MICE
            imputation, for a single train-test split."""
        X_train, _, X_test, _ = self._split(split_i)
        self._find_missing_indices(split_i, X_train, X_test)
        train_cat = X_train[self.cat_vars]
        self._scalers[split_i], self._imputers[split_i] = {}, {}
        for mice_imp_i in range(self.swm.n_mice_imputations):
            self._single_mice_imp(split_i, mice_imp_i, train_cat)

    def _find_missing_indices(self, split_i: int,
                              X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Find the indices (for this train-test split) of the missing values
            of every non-binary categorical variable. These missing values are
            consistent across MICE imputation as they weren't imputed using
            MICE. We find these indices for the train and test folds, although
            the test-fold indices aren't used until later imputation."""
        fold_dfs = {'train': X_train, 'test': X_test}
        for fold, df in fold_dfs.items():
            self.missing_i[fold][split_i] = find_missing_indices(
                df[self.cat_vars])

    def _single_mice_imp(self, split_i: int, mice_imp_i: int,
                         train_cat: pd.DataFrame):
        """Fit imputers for every non-binary categorical variable, for a single
            MICE imputation, in a single train-test split."""
        train_cont_bin = self.swm.get_imputed_X_df(split_i, 'train', mice_imp_i)
        self._fit_scalers(split_i, mice_imp_i, train_cont_bin)
        train_cont_bin = self._scale(split_i, mice_imp_i, train_cont_bin)
        self._imputers[split_i][mice_imp_i] = {}
        for cat_var_name in self.cat_vars:
            self._fit_imputer(split_i, mice_imp_i, train_cont_bin,
                              train_cat[cat_var_name])

    def _fit_scalers(self, split_i: int,
                     mice_imp_i: int, train_cont_bin: pd.DataFrame):
        """Fit scalers for continuous features. We need to scale them in order
            to use fast solvers for multinomial logistic regression in
            sklearn."""
        self._scalers[split_i][mice_imp_i] = RobustScaler()
        self._scalers[split_i][mice_imp_i].fit(
            train_cont_bin.loc[:, self.swm.cont_vars].values)

    def _scale(self, split_i: int, mice_imp_i: int,
               X: pd.DataFrame) -> pd.DataFrame:
        """Scale continuous features."""
        X = X.copy()
        X.loc[:, self.swm.cont_vars] = self._scalers[split_i][
            mice_imp_i].transform(X.loc[:, self.swm.cont_vars].values)
        return X

    def _fit_imputer(self, split_i: int, mice_imp_i: int,
                     train_cont_bin: pd.DataFrame, cat_var: pd.Series):
        """Fit imputation model of a single non-binary categorical variable, in
            a single MICE imputation, in a single train-test split. We only fit
            the model if there is at least one missing value of cat_var in the
            train or test fold (otherwise there would be nothing for the model
            to impute later)

        Args:
            split_i: Index of the train-test split
            mice_imp_i: Index of the MICE imputation within this TTS
            train_cont_bin: The (partly MICE-imputed) continuous and binary
                variables used as the imputation model's features
            cat_var: The non-binary categorical variable being modelled
        """
        if any((self.missing_i['train'][split_i][cat_var.name].shape[0],
                self.missing_i['test'][split_i][cat_var.name].shape[0])):
            obs_i = train_cont_bin.index.difference(
                self.missing_i['train'][split_i][cat_var.name])
            self._imputers[split_i][mice_imp_i][cat_var.name] = (
                LogisticRegression(penalty='none', solver='sag', max_iter=3000,
                                   multi_class='multinomial', n_jobs=-1,
                                   random_state=self.random_seed))
            self._imputers[split_i][mice_imp_i][cat_var.name].fit(
                train_cont_bin.iloc[obs_i].values, cat_var.iloc[obs_i].values)

    def impute_X_df(self, fold: str, split_i: int, mice_imp_i: int,
                    cat_imp_i: int) -> pd.DataFrame:
        """Imputes missing values of the non-binary categorical variables for a
            given MICE iteration, for a given fold, from a given train-test
            split. Constructs DataFrame (no missing values) containing all the
            categorical and continuous variables, except lactate and albumin,
            and the albumin and lactate missingness indicators. Uses a
            pseudo-index to allow different imputations (e.g. multiple
            categorical imputations per MICE imputation) to be reproduced.

        Args:
            split_i: Train-test split iteration index
            fold: in ('train', 'test')
            mice_imp_i: MICE imputation index (for this train-test split)
            cat_imp_i: Pseudo index for this categorical imputation

        Returns:
            Columns are (self.cat_vars + self.swm.cont_vars +
                         self.swm.binary_vars)
        """
        cont_bin_df = self.swm.get_imputed_X_df(split_i, fold, mice_imp_i)
        cont_bin_df_scaled = self._scale(split_i, mice_imp_i,
                                         cont_bin_df)
        cat_df = self._get_categorical_df(split_i, fold)
        for cat_var_name in self.cat_vars:
            cat_df = self._impute_cat_var(split_i, mice_imp_i, fold,
                                          cat_var_name, cont_bin_df_scaled,
                                          cat_df, cat_imp_i)
        return cat_df.join(cont_bin_df)

    def _get_categorical_df(self, split_i: int, fold: str) -> pd.DataFrame:
        if fold == 'train':
            X_df, _, _, _ = self._split(split_i)
        elif fold == 'test':
            _, _, X_df, _ = self._split(split_i)
        return X_df[self.cat_vars]

    def _impute_cat_var(self, split_i: int, mice_imp_i: int, fold: str,
                        cat_var_name: str, cont_bin_df_scaled: pd.DataFrame,
                        cat_df: pd.DataFrame, cat_imp_i: int) -> pd.DataFrame:
        """pred_probs is of shape (n_missing_values, n_classes), where each row
            corresponds to a missing value of cat_var_name, and each columns is
            the predicted probability that the missing value is that class.
            Rather than imputing each missing value using idxmax(pred_probs), we
            impute each missing value probabilistically using pred_probs."""
        missing_i = self.missing_i[fold][split_i][cat_var_name]
        if missing_i.shape[0]:
            imputer = self._imputers[split_i][mice_imp_i][cat_var_name]
            pred_probs = imputer.predict_proba(
                cont_bin_df_scaled.loc[missing_i].values)
            rnd = np.random.RandomState(self.random_seed + cat_imp_i)
            pred_classes = np.empty_like(missing_i)
            for i in range(missing_i.shape[0]):
                pred_classes[i] = rnd.choice(imputer.classes_,
                                             p=pred_probs[i, :])
            cat_df.loc[missing_i, cat_var_name] = pred_classes
        return cat_df


class LactateAlbuminImputer(Splitter):
    """Impute missing values of target (lactate or albumin)."""

    def __init__(self,
                 df: pd.DataFrame,
                 categorical_imputer: CategoricalImputer,
                 imputation_target: str,
                 imputation_model_factory: Callable[
                     [pd.Index, Dict[str, Tuple], str], GAM],
                 winsor_quantiles: Tuple[float, float],
                 transformer: Type[
                     Union[GammaTransformer, QuantileTransformer]],
                 transformer_args: Dict[str, Any],
                 multi_cat_vars: Dict[str, Tuple],
                 indication_var_name: str):
        """
        Args:
            df: Must just contain the variable to impute, plus the mortality
                variable (latter needed for compatibility with Splitter).
            categorical_imputer: With pre-fit imputers for all categorical
                variables
            imputation_target: Name of lactate or albumin variable
            imputation_model_factory: Function which returns specified (but not
                yet fitted) models of the transformed imputation target
            winsor_quantiles: Lower and upper quantiles to winsorize
                continuous variables at by default
            transformer: class that will be used to transform
                imputation_target
            transformer_args: arguments to pass to transformer class on
                instantiation
            multi_cat_vars: Keys are non-binary discrete variables, values are
                the categories (excluding null values) prior to integer encoding
            indication_var_name: Name of the indication column
        """
        super().__init__(df, categorical_imputer.tts,
                         categorical_imputer.target_variable_name)
        self.cat_imputer = categorical_imputer
        self.imp_target = imputation_target
        self.model_factory = imputation_model_factory
        self.winsor_quantiles = winsor_quantiles
        self.trans = transformer
        self.trans_args = transformer_args
        self.multi_cat_vars = multi_cat_vars
        self.ind_var_name = indication_var_name
        self._check_df(df)
        self.missing_i: Dict[str,  # fold name
                             Dict[int,  # train-test split index
                                  np.ndarray]] = {'train': {}, 'test': {}}
        self._winsor_thresholds: Dict[int,  # train-test split index
                                      Tuple[float, float]] = {}
        self._transformers: Dict[
            int,  # train-test split index
            Union[GammaTransformer, QuantileTransformer]] = {}
        self._imputers: Dict[int,  # train-test split index
                             GAM] = {}

    def _check_df(self, df: pd.DataFrame):
        """Check that passed DataFrame has correct columns, and no others."""
        assert len(df.columns) == 2
        for col in (self.target_variable_name, self.imp_target):
            assert col in df.columns

    def fit(self):
        """Fit albumin/lactate imputation models for every train-test split."""
        for i in pb(range(self.tts.n_splits), prefix='Split iteration'):
            self._single_train_test_split(i)

    def _single_train_test_split(self, split_i: int):
        """Fit target (albumin or lactate) imputation models for a single
            train-test split. target_train and target_test are DataFrames with
            a single column."""
        target_train, _, target_test, _ = self._split(split_i)
        for X in (target_train, target_test):  # TODO: Remove these 2 lines
            assert isinstance(X, pd.DataFrame)
        self._find_missing_indices(split_i, target_train, target_test)
        obs_target_train = self._get_observed_values(
            'train', split_i, target_train).values
        obs_target_train = self._winsorize(split_i, obs_target_train)
        obs_target_train = self._fit_transform(split_i, obs_target_train)
        self._fit_combine_gams(split_i, obs_target_train)

    def _find_missing_indices(self, split_i: int, target_train: pd.DataFrame,
                              target_test: pd.DataFrame):
        """Find the indices (for this train-test split) of the missing values
            of the imputation target. We find these indices for the train and
            test folds, although the test-fold indices aren't used until later
            imputation."""
        fold_dfs = {'train': target_train, 'test': target_test}
        for fold, df in fold_dfs.items():
            missing_i = find_missing_indices(df)
            self.missing_i[fold][split_i] = missing_i[self.imp_target]

    def _get_observed_values(self, fold: str, split_i: int,
                             X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[X.index.difference(self.missing_i[fold][split_i])]

    def _winsorize(self, split_i: int, target: pd.DataFrame) -> pd.DataFrame:
        """Winsorizes the only column in X. Also fits winsor_thresholds for this
            train-test split if not already fit (this fitting should happen on
            the train fold)."""
        try:
            target, _ = winsorize_novel(target, thresholds={
                self.imp_target: self._winsor_thresholds[split_i]})
        except KeyError:
            target, w_thresholds = winsorize_novel(
                target, cont_vars=[self.imp_target],
                quantiles=self.winsor_quantiles)
            self._winsor_thresholds[split_i] = w_thresholds[self.imp_target]
        return target

    def _fit_transform(self, split_i: int,
                       target_train: pd.DataFrame) -> pd.DataFrame:
        self._transformers[split_i] = self.trans(**self.trans_args)
        self._transformers[split_i].fit(target_train.values)
        target_train[self.imp_target] = self._transformers[split_i].transform(
            target_train.values)
        return target_train

    def _fit_combine_gams(self, split_i: int, obs_target_train: pd.DataFrame):
        gams = []
        for mice_imp_i in range(self.cat_imputer.swm.n_mice_imputations):
            for cat_imp_i in range(self.cat_imputer.imp_multiple):
                features_train = self.cat_imputer.impute_X_df(
                    'train', split_i, mice_imp_i, cat_imp_i)
                obs_features_train = self._get_observed_values(
                    'train', split_i, features_train)
                gam = self.model_factory(obs_features_train.columns,
                                         self.multi_cat_vars,
                                         self.ind_var_name)
                gam.fit(obs_features_train.values, obs_target_train.values)
                gams.append(gam)
        self._imputers[split_i] = combine_mi_gams(gams)

    def impute(self, fold: str, split_i: int, mice_imp_i: int, cat_imp_i: int,
               lac_alb_imp_i: int) -> pd.Series:
        # TODO: Remember to inverse transform imputer's predictions
        raise NotImplementedError


# TODO: Class which takes CategoricalImputer and LactateImputer / AlbuminImputer
#   for and yields complete X DataFrames plus mortality labels


# class ContImpPreprocessor:
#     """Prepares output from notebook 4 (list of DataFrames from MICE and
#         categorical imputation) for input to GAMs for imputation of
#         remaining unimputed variables, e.g. lactate, albumin."""
#
#     def __init__(self,
#                  imp_dfs: List[pd.DataFrame],
#                  target: str,
#                  drop_vars: List[str]):
#         """Args:
#             imp_dfs: Output of imputation in notebook 4. Each DataFrame
#                 contains complete columns for all variables except a select
#                 few that have missing values and will be imputed in production
#                 if missing, e.g. lactate, albumin.
#             target: The variable from imp_dfs for imputation. The indices of
#                 the missing values for target are assumed to be consistent
#                 across imp_dfs
#             drop_vars: Variables from imp_dfs not to include in the imputation
#                 models
#         """
#         self.imp_dfs = copy.deepcopy(imp_dfs)
#         self.target = target
#         self.drop_vars = drop_vars
#         self._i = None
#         self.X = {'train': [], 'missing': []}
#         self.y_train = None
#         self.y_train_trans = None
#
#     @property
#     def n_imp_dfs(self) -> int:
#         return len(self.imp_dfs)
#
#     def preprocess(self):
#         self._i = self._get_train_missing_i()
#         self._y_train_split()
#         for i in pb(range(self.n_imp_dfs)):
#             self._drop_unused_variables(i)
#             self._X_train_missing_split(i)
#
#     def yield_train_X_y(self, i, trans_y=False):
#         """Optionally yields a transformed version of y (this
#             transformation currently has to occur by some
#             method external to this class."""
#         if trans_y:
#             return self.X['train'][i].values, self.y_train_trans
#         else:
#             return self.X['train'][i].values, self.y_train
#
#     def _get_train_missing_i(self) -> Dict[str, pd.Int64Index]:
#         return {'train': self.imp_dfs[0].loc[self.imp_dfs[0][
#             self.target].notnull()].index,
#                 'missing': self.imp_dfs[0].loc[self.imp_dfs[0][
#                     self.target].isnull()].index}
#
#     def _drop_unused_variables(self, i: int):
#         self.imp_dfs[i] = self.imp_dfs[i].drop(self.drop_vars, axis=1)
#
#     def _y_train_split(self):
#         """y hasn't been imputed yet, so should be the same in all
#             imp_dfs."""
#         self.y_train = copy.deepcopy(
#             self.imp_dfs[0].loc[self._i['train'], self.target].values)
#         for i in range(1, self.n_imp_dfs):  # Check that all y are same
#             assert ((self.y_train ==
#                      self.imp_dfs[i].loc[self._i['train'],
#                                          self.target].values).all())
#
#     def _X_train_missing_split(self, i: int):
#         for fold in self.X.keys():
#             self.X[fold].append(self.imp_dfs[i].loc[
#                                     self._i[fold]].drop(self.target,
#                                                         axis=1).copy())
