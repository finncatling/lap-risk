import copy
from typing import List, Dict, Union, Tuple, Any, Callable, Type

import numpy as np
import pandas as pd
from progressbar import progressbar as pb
from pygam import GAM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from statsmodels.discrete import discrete_model
from statsmodels.imputation.mice import MICEData
from statsmodels.regression import linear_model

from utils.gam import combine_mi_gams
from utils.model.albumin import GammaTransformer
from utils.model.novel import winsorize_novel, NOVEL_MODEL_VARS
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
        """Add information about an imputation stage, and calculate the number	
            of imputations it will require.	
        Args:	
            description: Description of this imputation stage	
            df: Containing variables to be imputed during this stage. May also	
                contain other variables as long as they have no missing values
        """
        self.descriptions.append(description)
        self._determine_adjusted_n_imputations(df)

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

    def _split_then_join_Xy(self, i: int) -> (pd.DataFrame, pd.DataFrame):
        """Thin wrapper around ._split() which adds y_train (the mortality
            labels) back into X_train, and adds y_test back into X_test.
            This is convenient for input to the MICE and categorical imputation
            models, which use the target as a feature."""
        train, y_train, test, y_test = self._split(i)
        train[self.target_variable_name] = y_train
        test[self.target_variable_name] = y_test
        return train, test

    def get_imputed_variables(
        self, split_i: int, fold_name: str, imp_i: int
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
        for var_name, missing_i in self.missing_i[fold_name][split_i].items():
            imp_df.iloc[missing_i, imp_df.columns.get_loc(var_name)] = (
                self.imputed[fold_name][split_i][imp_i][var_name]
            )
        return imp_df


class SplitterWinsorMICE(Imputer):
    """Performs winsorization then MICE for each predefined train-test split.
        MICE is limited to the continuous variables and binary variables
        (except those related to lactate and albumin). For efficiency, we store
        only the imputed values and their indices.

        Note that we use the target variable (mortality) as a feature in this
        imputation model. Our earlier code (in data_check.py) guarantees that
        the target variable is complete, so no values are imputed for it."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_test_splitter: TrainTestSplitter,
        target_variable_name: str,
        cont_variables: List[str],
        binary_variables: List[str],
        winsor_quantiles: Tuple[float, float],
        winsor_include: Dict[str, Tuple[bool, bool]],
        n_mice_imputations: int,
        n_mice_burn_in: int,
        n_mice_skip: int,
        random_seed: int
    ):
        """
        Args:
            df: DataFrame containing the continuous variables and binary
                variables (except those related to lactate and albumin), and
                the target (mortality labels). This DataFrame still contains
                all its missing values, i.e. no imputation yet
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
                discarded between retained DataFrames
            random_seed: statsmodels MICE doesn't provide the option to pass in
                a random seed, so we use this to set a different global numpy
                random seed for each split
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
        self.random_seed = random_seed
        self.winsor_thresholds: Dict[int,  # train-test split index
                                     Dict[str,  # variable name
                                          Tuple[float, float]]] = {}

    @property
    def all_vars(self):
        return self.cont_vars + self.binary_vars + [self.target_variable_name]

    def _sanity_check_variables(self):
        """Check that df only contains the variables specified."""
        assert len(self.all_vars) == len(self.df.columns)
        for var in self.all_vars:
            assert var in self.df.columns

    def split_winsorize_mice(self):
        """Split df according to pre-defined train-test splits, perform
            winsorization (using thresholds from train fold to winsorize test
            fold), then run MICE for train and test folds of all train-test
            splits."""
        for split_i in pb(range(self.tts.n_splits), prefix="Split iteration"):
            train, test = self._split_then_join_Xy(split_i)
            train, test = self._winsorize(split_i, train, test)
            for fold_name, df in {
                "train": train,
                "test": test
            }.items():
                self._single_fold_mice(split_i, fold_name, df)

    def _winsorize(
        self, split_i: int, train: pd.DataFrame, test: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        train, winsor_thresholds = winsorize_novel(
            train,
            cont_vars=self.cont_vars,
            quantiles=self.winsor_quantiles,
            include=self.winsor_include,
        )
        test, _ = winsorize_novel(test, thresholds=winsor_thresholds)
        self.winsor_thresholds[split_i] = winsor_thresholds
        return train, test

    def _single_fold_mice(self, split_i: int, fold: str, df: pd.DataFrame):
        """Set up and run MICE for a single fold from a single train-test
            split."""
        np.random.seed(self.random_seed + split_i)
        mice_data = MICEData(df)
        self.missing_i[fold][split_i] = copy.deepcopy(mice_data.ix_miss)
        mice_data = self._set_mice_imputers(mice_data)
        self.imputed[fold][split_i] = {}  # dict will hold fold's imputed values
        self._run_mice_loop(split_i, fold, mice_data)

    def _set_mice_imputers(self, mice_data: MICEData) -> MICEData:
        """Although we set an imputer for the target variable, this variable is
            guaranteed complete by our earlier code so no values will be
            imputed."""
        for var in self.cont_vars:
            mice_data.set_imputer(
                var, model_class=linear_model.OLS, fit_kwds={"disp": False}
            )
        for var in self.binary_vars + [self.target_variable_name]:
            mice_data.set_imputer(
                var, model_class=discrete_model.Logit, fit_kwds={"disp": False}
            )
        return mice_data

    def _run_mice_loop(self, split_i: int, fold: str, mice_data: MICEData):
        """'Burn-in' and 'skip' imputations are discarded."""
        for _ in range(self.n_mice_burn_in):
            mice_data.update_all()
        for imputation_i in range(self.n_mice_imputations):
            if imputation_i:
                mice_data.update_all(self.n_mice_skip + 1)
            self._store_imputed(split_i, fold, imputation_i, mice_data.data)

    def _store_imputed(
        self, split_i: int, fold: str, imputation_i: int, imp_df: pd.DataFrame
    ):
        """Store just the imputed values from a single MICE iteration."""
        self.imputed[fold][split_i][imputation_i] = {}
        for var_name, missing_i in self.missing_i[fold][split_i].items():
            self.imputed[fold][split_i][imputation_i][var_name] = (
                imp_df.iloc[
                    missing_i, imp_df.columns.get_loc(var_name)].copy().values
            )


class CategoricalImputer(Imputer):
    """Imputes missing values of non-binary categorical variables, using
        output of earlier MICE."""

    def __init__(
        self,
        df: pd.DataFrame,
        splitter_winsor_mice: SplitterWinsorMICE,
        cat_vars: List[str],
        random_seed,
    ):
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
            random_seed: For reproducibility
        """
        super().__init__(
            df,
            splitter_winsor_mice.tts,
            splitter_winsor_mice.target_variable_name
        )
        self.swm = splitter_winsor_mice
        self.cat_vars = cat_vars
        self.random_seed = random_seed

    def impute(self):
        """Impute missing values for every non-binary categorical variable,
            in every MICE imputation, in every train-test split."""
        for i in pb(range(self.tts.n_splits), prefix="Split iteration"):
            self._single_train_test_split(i)

    def _single_train_test_split(self, split_i: int):
        """Impute missing values for every non-binary categorical variable,
            in every MICE imputation, for a single train-test split."""
        train, test = self._split_then_join_Xy(split_i)
        self._find_missing_indices(split_i, train, test)
        self._initialise_subdicts_for_imputation_storage(split_i)
        for mice_imp_i in range(self.swm.n_mice_imputations):
            self._single_mice_imp(
                split_i=split_i,
                mice_imp_i=mice_imp_i,
                train_cat_vars=train[self.cat_vars]
            )

    def _find_missing_indices(
        self, split_i: int, train: pd.DataFrame, test: pd.DataFrame
    ):
        """Find the indices (for this train-test split) of the missing values
            of every non-binary categorical variable. These missing values are
            consistent across MICE imputations. We find these indices for the
            train and test folds, although the test-fold indices aren't used
            until later imputation."""
        for fold_name, df in {
            "train": train,
            "test": test
        }.items():
            self.missing_i[fold_name][split_i] = find_missing_indices(
                df[self.cat_vars]
            )

    def _initialise_subdicts_for_imputation_storage(self, split_i: int):
        for fold_name in self.imputed.keys():
            self.imputed[fold_name][split_i] = {}
            for mice_imp_i in range(self.swm.n_mice_imputations):
                self.imputed[fold_name][split_i][mice_imp_i] = {}

    def _single_mice_imp(
        self,
        split_i: int,
        mice_imp_i: int,
        train_cat_vars: pd.DataFrame,
    ):
        """Impute missing values for every non-binary categorical variable, for
            a single MICE imputation, in a single train-test split."""
        cont_bin_target_vars = {
            "train": self.swm.get_imputed_variables(
                split_i, "train", mice_imp_i
            ),
            "test": self.swm.get_imputed_variables(
                split_i, "test", mice_imp_i
            ),
        }
        cont_bin_target_vars = self._scale(cont_bin_target_vars)
        self._impute_all_cat_vars(
            split_i=split_i,
            mice_imp_i=mice_imp_i,
            imp_folds_features=cont_bin_target_vars,
            imp_train_targets=train_cat_vars
        )

    def _scale(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        scalers = self._fit_scalers(dfs["train"])
        for fold_name in dfs.keys():
            dfs[fold_name] = self._scale_fold(
                fold=dfs[fold_name],
                scalers=scalers
            )
        return dfs

    def _fit_scalers(self, train: pd.DataFrame) -> RobustScaler:
        """Fit scalers for continuous features. We need to scale them in order
            to use fast solvers for multinomial logistic regression in
            sklearn."""
        scalers = RobustScaler()
        scalers.fit(train.loc[:, self.swm.cont_vars].values)
        return scalers

    def _scale_fold(
        self,
        fold: pd.DataFrame,
        scalers: RobustScaler
    ) -> pd.DataFrame:
        """Scale continuous features."""
        fold.loc[
            :,
            self.swm.cont_vars
        ] = scalers.fit_transform(
            fold.loc[
                :,
                self.swm.cont_vars
            ].values
        )
        return fold

    def _impute_all_cat_vars(
        self,
        split_i: int,
        mice_imp_i: int,
        imp_folds_features: Dict[str, pd.DataFrame],
        imp_train_targets: pd.DataFrame,
    ):
        """Impute values for all non-binary categorical variables, in
            a single MICE imputation, in a single train-test split.

        Args:
            split_i: Index of the train-test split
            mice_imp_i: Index of the MICE imputation
            imp_folds_features: Keys are 'train' and 'test'. Values are the
                (partly MICE-imputed) continuous, binary and target variables
                from that fold, which will be used as the imputation model's
                features
            imp_train_targets: All non-binary categorical variables from the
                train fold
        """
        imputers = self._fit_imputers(
            split_i=split_i,
            imp_train_features=imp_folds_features["train"],
            imp_train_targets=imp_train_targets
        )
        for fold_name in imp_folds_features:
            for cat_var_name in imp_train_targets.columns:
                self._impute_single_cat_var(
                    split_i=split_i,
                    mice_imp_i=mice_imp_i,
                    fold_name=fold_name,
                    cat_var_name=cat_var_name,
                    imp_features=imp_folds_features[fold_name],
                    imputer=imputers[cat_var_name]
                )

    def _fit_imputers(
        self,
        split_i: int,
        imp_train_features: pd.DataFrame,
        imp_train_targets: pd.DataFrame,
    ) -> Dict[str, Union[LogisticRegression, None]]:
        """Fit imputation models for all non-binary categorical variables, in
            a single MICE imputation, in a single train-test split.

        Args:
            split_i: Index of the train-test split
            imp_train_features: The (partly MICE-imputed) continuous, binary and
                target variables used as the imputation model's features
            imp_train_targets: All non-binary categorical variables
        """
        imputers = {}
        for cat_var_name in self.cat_vars:
            imputers[cat_var_name] = self._fit_single_imputer(
                split_i=split_i,
                imp_train_features=imp_train_features,
                imp_train_target=imp_train_targets[cat_var_name]
            )
        return imputers

    def _fit_single_imputer(
        self,
        split_i: int,
        imp_train_features: pd.DataFrame,
        imp_train_target: pd.Series,
    ) -> Union[LogisticRegression, None]:
        """Fit imputation model of a single non-binary categorical variable, in
            a single MICE imputation, in a single train-test split. We only fit
            the model if there is at least one missing value of cat_var in the
            train or test fold (otherwise there would be nothing for the model
            to impute later). Returns None if imputation model not fit.

        Args:
            split_i: Index of the train-test split
            imp_train_features: The (partly MICE-imputed) continuous, binary and
                target variables used as the imputation model's features
            imp_train_target: The non-binary categorical variable being modelled
        """
        if any((
            self.missing_i["train"][split_i][imp_train_target.name].shape[0],
            self.missing_i["test"][split_i][imp_train_target.name].shape[0],
        )):
            obs_i = imp_train_features.index.difference(
                self.missing_i["train"][split_i][imp_train_target.name]
            )
            imputer = LogisticRegression(
                penalty="none",
                solver="sag",
                max_iter=3000,
                multi_class="multinomial",
                n_jobs=-1,
                random_state=self.random_seed,
            )
            imputer.fit(
                imp_train_features.iloc[obs_i].values,
                imp_train_target.iloc[obs_i].values
            )
            return imputer

    def _impute_single_cat_var(
        self,
        split_i: int,
        mice_imp_i: int,
        fold_name: str,
        cat_var_name: str,
        imp_features: pd.DataFrame,
        imputer: Union[LogisticRegression, None]
    ):
        """pred_probs is of shape (n_missing_values, n_classes), where each row
            corresponds to a missing value of cat_var_name, and each columns is
            the predicted probability that the missing value is that class.
            Rather than imputing each missing value using idxmax(pred_probs), we
            impute each missing value probabilistically using pred_probs."""
        missing_i = self.missing_i[fold_name][split_i][cat_var_name]
        if missing_i.shape[0]:
            pred_probs = imputer.predict_proba(
                imp_features.loc[missing_i].values
            )
            rnd = np.random.RandomState(self.random_seed + mice_imp_i)
            pred_classes = np.empty_like(missing_i)
            for i in range(missing_i.shape[0]):
                pred_classes[i] = rnd.choice(
                    imputer.classes_,
                    p=pred_probs[i, :]
                )
            self.imputed[fold_name][split_i][mice_imp_i][cat_var_name] = (
                pred_classes
            )

    def get_imputed_df(
        self, fold_name: str, split_i: int, imp_i: int
    ) -> pd.DataFrame:
        """Constructs DataFrame containing the continuous and binary variables
            (except those related to lactate and albumin), the target variable
            and the non-binary categorical variables, including their imputed
            missing values for a given fold, train-test split and imputation
            (MICE and categorical imputation) iteration."""
        cont_bin_target_df = self.swm.get_imputed_variables(
            split_i=split_i, fold_name=fold_name, imp_i=imp_i
        )
        cat_df = self.get_imputed_variables(
            split_i=split_i, fold_name=fold_name, imp_i=imp_i
        )
        return cont_bin_target_df.join(cat_df)


class LactateAlbuminImputer(Imputer):
    """Impute missing values of target (lactate or albumin)."""

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_imputer: CategoricalImputer,
        imputation_target: str,
        imputation_model_factory: Callable[
            [pd.Index, Dict[str, Tuple], str], GAM],
        winsor_quantiles: Tuple[float, float],
        transformer: Type[Union[GammaTransformer, QuantileTransformer]],
        transformer_args: Dict[str, Any],
        multi_cat_vars: Dict[str, Tuple],
        indication_var_name: str,
    ):
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
        super().__init__(
            df, categorical_imputer.tts,
            categorical_imputer.target_variable_name
        )
        self.cat_imputer = categorical_imputer
        self.cont_vars = NOVEL_MODEL_VARS["cont"]
        self.imp_target = imputation_target
        self.model_factory = imputation_model_factory
        self.winsor_quantiles = winsor_quantiles
        self.trans = transformer
        self.trans_args = transformer_args
        self.multi_cat_vars = multi_cat_vars
        self.ind_var_name = indication_var_name
        self._check_df(df)
        self.missing_i: Dict[
            str, Dict[int, np.ndarray]  # fold name  # train-test split index
        ] = {"train": {}, "test": {}}
        self._winsor_thresholds: Dict[
            int, Tuple[float, float]  # train-test split index
        ] = {}
        self._transformers: Dict[
            int, Union[GammaTransformer, QuantileTransformer]
            # train-test split index
        ] = {}
        self._imputers: Dict[int, GAM] = {}  # train-test split index

    def _check_df(self, df: pd.DataFrame):
        """Check that passed DataFrame has correct columns, and no others."""
        assert len(df.columns) == 2
        for col in (self.target_variable_name, self.imp_target):
            assert col in df.columns

    def fit(self):
        """Fit albumin/lactate imputation models for every train-test split."""
        for i in pb(range(self.tts.n_splits), prefix="Split iteration"):
            self._single_train_test_split(i)

    def _single_train_test_split(self, split_i: int):
        """Fit target (albumin or lactate) imputation models for a single
            train-test split. target_train and target_test are DataFrames with
            a single column."""
        target_train, _, target_test, _ = self._split(split_i)
        for X in (target_train, target_test):  # TODO: Remove these 2 lines
            assert isinstance(X, pd.DataFrame)
        self._find_missing_indices(split_i, target_train, target_test)
        obs_target_train = self._get_observed_values("train", split_i,
                                                     target_train)
        obs_target_train = self._winsorize(split_i, obs_target_train)
        obs_target_train = self._fit_transform(split_i, obs_target_train)
        self._fit_combine_gams(split_i, obs_target_train)

    def _find_missing_indices(
        self, split_i: int, target_train: pd.DataFrame,
        target_test: pd.DataFrame
    ):
        """Find the indices (for this train-test split) of the missing values
            of the imputation target. We find these indices for the train and
            test folds, although the test-fold indices aren't used until later
            imputation."""
        fold_dfs = {"train": target_train, "test": target_test}
        for fold, df in fold_dfs.items():
            missing_i = find_missing_indices(df)
            self.missing_i[fold][split_i] = missing_i[self.imp_target]

    def _get_observed_values(
        self, fold: str, split_i: int, X: pd.DataFrame
    ) -> pd.DataFrame:
        return X.loc[X.index.difference(self.missing_i[fold][split_i])]

    def _winsorize(self, split_i: int, target: pd.DataFrame) -> pd.DataFrame:
        """Winsorizes the only column in X. Also fits winsor_thresholds for this
            train-test split if not already fit (this fitting should happen on
            the train fold)."""
        try:
            target, _ = winsorize_novel(
                target,
                thresholds={self.imp_target: self._winsor_thresholds[split_i]}
            )
        except KeyError:
            target, w_thresholds = winsorize_novel(
                target,
                cont_vars=list(self.cont_vars),
                quantiles=self.winsor_quantiles
            )
            self._winsor_thresholds[split_i] = w_thresholds[self.imp_target]
        return target

    def _fit_transform(self, split_i: int,
                       target_train: pd.DataFrame) -> pd.DataFrame:
        self._transformers[split_i] = self.trans(**self.trans_args)
        self._transformers[split_i].fit(target_train.values)
        target_train[self.imp_target] = self._transformers[split_i].transform(
            target_train.values
        )
        return target_train

    def _fit_combine_gams(self, split_i: int, obs_target_train: pd.DataFrame):
        gams = []
        # TODO: Update this nested loop now that we are doing only one earlier
        #   stage of imputation
        for mice_imp_i in range(self.cat_imputer.swm.n_mice_imputations):
            for cat_imp_i in range(self.cat_imputer.imp_multiple):
                features_train = self.cat_imputer.impute_X_df(
                    "train", split_i, mice_imp_i, cat_imp_i
                )
                obs_features_train = self._get_observed_values(
                    "train", split_i, features_train
                )
                gam = self.model_factory(
                    obs_features_train.columns, self.multi_cat_vars,
                    self.ind_var_name
                )
                gam.fit(obs_features_train.values, obs_target_train.values)
                gams.append(gam)
        self._imputers[split_i] = combine_mi_gams(gams)

    def impute(
        self,
        fold: str,
        split_i: int,
        mice_imp_i: int,
        cat_imp_i: int,
        lac_alb_imp_i: int,
    ) -> pd.Series:
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
