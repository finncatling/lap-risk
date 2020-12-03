import copy
import operator
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
import pandas as pd
from progressbar import progressbar as pb
from pygam import LogisticGAM, s, f, te, LinearGAM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from statsmodels.discrete import discrete_model
from statsmodels.imputation.mice import MICEData
from statsmodels.regression import linear_model

from utils.gam import combine_mi_gams, quick_sample
from utils.impute import Imputer
from utils.indications import ohe_to_single_column
from utils.split import TrainTestSplitter

NOVEL_MODEL_VARS = {
    "cat": (
        "S03ASAScore",
        "S03CardiacSigns",
        "S03RespiratorySigns",
        "S03DiagnosedMalignancy",
        "S03Pred_Peritsoil",
        "S02PreOpCTPerformed",
        "S03ECG",
    ),
    "cont": (
        "S01AgeOnArrival",
        "S03SerumCreatinine",
        "S03PreOpArterialBloodLactate",
        "S03PreOpLowestAlbumin",
        "S03Sodium",
        "S03Potassium",
        "S03Urea",
        "S03WhiteCellCount",
        "S03Pulse",
        "S03SystolicBloodPressure",
        "S03GlasgowComaScore",
    ),
    # plus Indications variable
    "target": "Target",
}

# Need to add levels for the Indications variable below, once they are derived
MULTI_CATEGORY_LEVELS = {
    "S03ASAScore": (1.0, 2.0, 3.0, 4.0, 5.0),
    "S03CardiacSigns": (1.0, 2.0, 4.0, 8.0),
    "S03RespiratorySigns": (1.0, 2.0, 4.0, 8.0),
    "S03DiagnosedMalignancy": (1.0, 2.0, 4.0, 8.0),
    "S03Pred_Peritsoil": (1.0, 2.0, 4.0, 8.0),
}

WINSOR_QUANTILES = (0.001, 0.999)

LACTATE_VAR_NAME = "S03PreOpArterialBloodLactate"
ALBUMIN_VAR_NAME = "S03PreOpLowestAlbumin"
MISSINGNESS_SUFFIX = "_missing"


def novel_model_factory(
    columns: pd.Index,
    multi_cat_levels: Dict[str, Tuple],
    indication_var_name: str
) -> LogisticGAM:
    return LogisticGAM(
        s(
            columns.get_loc("S01AgeOnArrival"),
            n_splines=10,
            spline_order=2,
            lam=15
        )
        + s(
            columns.get_loc("S03SystolicBloodPressure"),
            n_splines=10,
            spline_order=2,
            lam=25
        )
        + te(
            columns.get_loc("S03Pulse"),
            columns.get_loc("S03ECG"),
            lam=(25, 2),
            n_splines=(10, 2),
            spline_order=(2, 0),
            dtype=("numerical", "categorical"),
        )
        + s(
            columns.get_loc("S03WhiteCellCount"),
            n_splines=10,
            spline_order=2,
            lam=15
        )
        + s(
            columns.get_loc("S03Sodium"),
            n_splines=10,
            spline_order=2,
            lam=25
        )
        + s(
            columns.get_loc("S03Potassium"),
            n_splines=10,
            spline_order=2,
            lam=25
        )
        + s(
            columns.get_loc(LACTATE_VAR_NAME),
            n_splines=10,
            spline_order=2,
            lam=15
        )
        + s(
            columns.get_loc(ALBUMIN_VAR_NAME),
            n_splines=10,
            spline_order=2,
            lam=25
        )
        + s(
            columns.get_loc("S03GlasgowComaScore"),
            spline_order=0,
            n_splines=13,
            lam=85
        )
        + f(columns.get_loc("S03ASAScore"), coding="dummy", lam=5)
        + f(
            columns.get_loc(f"{LACTATE_VAR_NAME}{MISSINGNESS_SUFFIX}"),
            coding="dummy",
            lam=5
        )
        + f(
            columns.get_loc(f"{ALBUMIN_VAR_NAME}{MISSINGNESS_SUFFIX}"),
            coding="dummy",
            lam=5
        )
        + te(
            columns.get_loc("S03DiagnosedMalignancy"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(5, 2),
            n_splines=(len(multi_cat_levels["S03DiagnosedMalignancy"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03Pred_Peritsoil"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(5, 2),
            n_splines=(len(multi_cat_levels["S03Pred_Peritsoil"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03CardiacSigns"),
            columns.get_loc("S03RespiratorySigns"),
            lam=8,
            n_splines=(
                len(multi_cat_levels["S03CardiacSigns"]),
                len(multi_cat_levels["S03RespiratorySigns"])
            ),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03SerumCreatinine"),
            columns.get_loc("S03Urea"),
            spline_order=(2, 2),
            lam=40.0,
            dtype=("numerical", "numerical"),
        )
        + te(
            columns.get_loc(indication_var_name),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(5, 2),
            # subtract 1 to account for missing indication category
            n_splines=(len(multi_cat_levels[indication_var_name]) - 1, 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
    )


def combine_categories(
    df: pd.DataFrame, category_mapping: Dict[str, Dict[float, float]]
) -> pd.DataFrame:
    """Combines values of categorical variables. Propagates missing values.
        each key-value pair in combine specifies a remapping of current
        categories to new ones. An example key-value pair in combine is
        'S03ECG' : {1.0: 0.0, 4.0: 1.0, 8.0: 1.0} which combines
        the two 'abnormal ecg' categories (4.0 and 8.0) together."""
    drop = []

    for v, mapping in category_mapping.items():
        temp_variable_name = f"{v}_temp"
        drop.append(temp_variable_name)
        df[temp_variable_name] = df[v].copy()
        df[v] = np.nan

        for old, new in mapping.items():
            df.loc[df[temp_variable_name] == old, v] = new

    return df.drop(drop, axis=1)


def label_encode(
    df: pd.DataFrame,
    multi_cat_levels: Dict[str, Tuple],
    missing_indication_value: str
) -> pd.DataFrame:
    """Encode labels for each novel-model categorical variable as integers, with
        missingness support."""
    for c, levels in multi_cat_levels.items():
        if c != "Indication":
            df[c] = df[c].astype(float)
            df[c] = [np.nan if np.isnan(x) else levels.index(x) for x in
                     df[c].values]
            df[c] = df[c].astype(float)
        else:
            df[c] = [
                np.nan if x == missing_indication_value else levels.index(x)
                for x in df[c].values
            ]
    return df


def winsorize_novel(
    df: pd.DataFrame,
    thresholds: Union[Dict[str, Tuple[float, float]], None] = None,
    cont_vars: Union[List[str], None] = None,
    quantiles: Union[Tuple[float, float], None] = None,
    include: Union[Dict[str, Tuple[bool, bool]], None] = None,
) -> (pd.DataFrame, Dict[str, Tuple[float, float]]):
    """Winsorize continuous variables at thresholds in thresholds_dict, or at
        specified quantiles if thresholds_dict is None. If thresholds_dict is
        None, upper and/or lower Winsorization for selected variables can be
        disabled using the include dict. Variables not specified in the include
        dict have Winsorization applied at upper and lower thresholds by
        default."""
    if thresholds is None:
        assert quantiles is not None
        assert cont_vars is not None
    else:
        assert quantiles is None
        assert cont_vars is None
        assert include is None

    if include is not None:
        for v in include.keys():
            assert v in cont_vars

    df = df.copy()
    ops = (operator.lt, operator.gt)
    if thresholds:
        for v, thresholds_tuple in thresholds.items():
            for i, threshold in enumerate(thresholds_tuple):
                if threshold is not None:
                    df.loc[ops[i](df[v], threshold), v] = threshold
    else:
        thresholds = {}
        for v in cont_vars:
            thresholds[v] = list(df[v].quantile(quantiles))
            for i, threshold in enumerate(thresholds[v]):
                try:
                    if include[v][i]:
                        df.loc[ops[i](df[v], threshold), v] = threshold
                    else:
                        thresholds[v][i] = None
                except (KeyError, TypeError):
                    df.loc[ops[i](df[v], threshold), v] = threshold

    return df, thresholds


def preprocess_novel_pre_split(
    df: pd.DataFrame,
    category_mapping: Dict[str, Dict[float, float]],
    indication_variable_name: str,
    indications: List[str],
    missing_indication_value: str,
    multi_category_levels: Dict[str, Tuple],
) -> pd.DataFrame:
    """In preparation for later data input to the novel model, does the data
        preprocessing steps which can be safely performed before train-test
        splitting."""
    df = df.copy()
    df = combine_categories(df, category_mapping)
    df = ohe_to_single_column(df, indication_variable_name, indications)
    df = label_encode(df, multi_category_levels, missing_indication_value)
    return df


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
        winsor_include: Union[Dict[str, Tuple[bool, bool]], None],
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
        self._sanity_check()
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

    def _sanity_check(self):
        self._sanity_check_variables()
        self._sanity_check_missingness()

    def _sanity_check_variables(self):
        """Check that df only contains the variables specified."""
        assert len(self.all_vars) == len(self.df.columns)
        for var in self.all_vars:
            assert var in self.df.columns

    def _sanity_check_missingness(self):
        """Checking that there are no rows in self.df where all values are
            missing (these cases would be dropped by statsmodels MICEData,
            which could create problems with the post-imputation data
            reconstruction)"""
        assert self.df.shape[0] == self.df.dropna(axis=0, how="all").shape[0]

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
                    missing_i,
                    imp_df.columns.get_loc(var_name)
                ].copy().values
            )

    def winsorize_after_get_imputed_variables(
        self, fold_name: str, split_i: int, imp_i: int
    ) -> pd.DataFrame:
        """Extends .get_imputed_variables() in base class with winsorization."""
        imp_df, _ = winsorize_novel(
            self.get_imputed_variables(fold_name, split_i, imp_i),
            thresholds=self.winsor_thresholds[split_i]
        )
        return imp_df

    def _find_missing_indices(
        self,
        split_i: int,
        train: pd.DataFrame,
        test: pd.DataFrame,
        variable_names: List[str]
    ):
        raise NotImplementedError


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
        self._find_missing_indices(split_i, train, test, self.cat_vars)
        self._initialise_subdicts_for_imputation_storage(split_i)
        for mice_imp_i in range(self.swm.n_mice_imputations):
            self._single_mice_imp(
                split_i=split_i,
                mice_imp_i=mice_imp_i,
                train_cat_vars=train[self.cat_vars]
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
                "train", split_i, mice_imp_i
            ),
            "test": self.swm.get_imputed_variables(
                "test", split_i, mice_imp_i
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
        fold.loc[:, self.swm.cont_vars] = scalers.fit_transform(
            fold.loc[:, self.swm.cont_vars].values
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
        cont_bin_target_df = self.swm.winsorize_after_get_imputed_variables(
            fold_name=fold_name,
            split_i=split_i,
            imp_i=imp_i
        )
        cat_df = self.get_imputed_variables(
            fold_name=fold_name,
            split_i=split_i,
            imp_i=imp_i
        )
        return cont_bin_target_df.join(cat_df)


class LactateAlbuminImputer(Imputer):
    """Impute missing values of lactate or albumin. There is no simple way
        to random seed GAM model fitting so imputation models will be different
        on each training iteration."""

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_imputer: CategoricalImputer,
        lacalb_variable_name: str,
        imputation_model_factory: Callable[
            [pd.Index, Dict[str, Tuple], str, bool], LinearGAM],
        winsor_quantiles: Tuple[float, float],
        multi_cat_vars: Dict[str, Tuple],
        indication_var_name: str,
        mortality_as_feature: bool,
        random_seed):
        """
        Args:
            df: Must just contain the variable to impute, plus the mortality
                variable (latter needed for compatibility with Splitter).
            categorical_imputer: With pre-fit imputers for all categorical
                variables
            lacalb_variable_name: Name of lactate or albumin variable
            imputation_model_factory: Function which returns specified (but not
                yet fitted) models of the transformed imputation target
            winsor_quantiles: Lower and upper quantiles to winsorize
                continuous variables at by default
            multi_cat_vars: Keys are non-binary discrete variables, values are
                the categories (excluding null values) prior to integer encoding
            indication_var_name: Name of the indication column
            mortality_as_feature: If True, uses mortality labels as a feature
                in this lactate / albumin imputation model (providing that
                mortality is a feature in the GAM specification in
                imputation_model_factory)
            random_seed: Used for QuantileTransformer
        """
        super().__init__(
            df,
            categorical_imputer.tts,
            categorical_imputer.target_variable_name
        )
        self.cat_imputer = categorical_imputer
        self.cont_vars = NOVEL_MODEL_VARS["cont"]  # TODO: Remove if unused?
        self.lacalb_variable_name = lacalb_variable_name
        self.model_factory = imputation_model_factory
        self.winsor_quantiles = winsor_quantiles
        self.multi_cat_vars = multi_cat_vars
        self.ind_var_name = indication_var_name
        self.mortality_as_feature = mortality_as_feature
        self.random_seed = random_seed
        self.imputed = None  # Override base class. This var shouldn't be used
        self._check_df(df)
        self.winsor_thresholds: Dict[
            int,  # train-test split index
            Tuple[float, float]
        ] = {}
        self.transformers: Dict[
            int,  # train-test split index
            QuantileTransformer
        ] = {}
        self.imputers: Dict[
            int,  # train-test split index
            LinearGAM
        ] = {}

    def _check_df(self, df: pd.DataFrame):
        """Check that passed DataFrame has correct columns, and no others."""
        assert set(df.columns) == {
            self.target_variable_name,
            self.lacalb_variable_name
        }

    def fit(self):
        """Fit albumin or lactate imputation models for every train-test
            split."""
        for i in pb(range(self.tts.n_splits), prefix="Split iteration"):
            self._single_train_test_split(i)

    def _single_train_test_split(self, split_i: int):
        """Fit albumin or lactate imputation models for a single train-test
            split. lacalb_train and lacalb_test are DataFrames with a single
            column of lactate / albumin values."""
        lacalb_train, _, lacalb_test, _ = self._split(split_i)
        self._find_missing_indices(
            split_i=split_i,
            train=lacalb_train,
            test=lacalb_test,
            variable_names=[self.lacalb_variable_name]
        )
        obs_lacalb_train = self._get_observed_values(
            fold="train",
            split_i=split_i,
            X=lacalb_train
        )
        obs_lacalb_train = self._winsorize(split_i, obs_lacalb_train)
        obs_lacalb_train = self._fit_transform(split_i, obs_lacalb_train)
        self._fit_combine_gams(split_i, obs_lacalb_train)

    def _get_observed_values(
        self, fold: str, split_i: int, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Note that the index of the returned DataFrame isn't reset."""
        return X.loc[X.index.difference(
            self.missing_i[fold][split_i][self.lacalb_variable_name]
        )]

    def _winsorize(
        self,
        split_i: int,
        lacalb: pd.DataFrame
    ) -> pd.DataFrame:
        """Winsorizes the only column in X. Also fits winsor_thresholds for this
            train-test split if not already fit (this fitting should happen on
            the train fold)."""
        try:
            lacalb, _ = winsorize_novel(
                df=lacalb,
                thresholds={
                    self.lacalb_variable_name: self.winsor_thresholds[split_i]
                }
            )
        except KeyError:
            lacalb, winsor_thresholds = winsorize_novel(
                df=lacalb,
                cont_vars=[self.lacalb_variable_name],
                quantiles=self.winsor_quantiles
            )
            self.winsor_thresholds[split_i] = winsor_thresholds[
                self.lacalb_variable_name
            ]
        return lacalb

    def _fit_transform(
        self,
        split_i: int,
        obs_lacalb_train: pd.DataFrame
    ) -> pd.DataFrame:
        """Note that, as lactate / albumin are effectively discretised in the
            NELA dataset (lactate is reported to 1 DP and albumin is reported
            to 0 DP), the resolution of QuantileTransformer's quantiles is
            limited despite us setting n_quantiles to a large number."""
        self.transformers[split_i] = QuantileTransformer(
            n_quantiles=10000,
            output_distribution='normal',
            random_state=self.random_seed
        )
        obs_lacalb_train[self.lacalb_variable_name] = self.transformers[
            split_i
        ].fit_transform(obs_lacalb_train.values)
        return obs_lacalb_train

    def _fit_combine_gams(
        self,
        split_i: int,
        obs_lacalb_train: pd.DataFrame
    ):
        gams = []
        for mice_imp_i in range(self.cat_imputer.swm.n_mice_imputations):
            features_train = self.cat_imputer.get_imputed_df(
                "train", split_i, mice_imp_i)
            if not self.mortality_as_feature:
                features_train = features_train.drop(
                    self.target_variable_name, axis=1)
            obs_features_train = self._get_observed_values(
                "train", split_i, features_train)
            gam = self.model_factory(
                obs_features_train.columns,
                self.multi_cat_vars,
                self.ind_var_name,
                self.mortality_as_feature)
            gam.fit(obs_features_train.values, obs_lacalb_train.values)
            gams.append(gam)
        self.imputers[split_i] = combine_mi_gams(gams)

    def impute(
        self,
        features: pd.DataFrame,
        split_i: int,
        lac_alb_imp_i: Union[int, None],
        probabilistic: bool
    ) -> np.ndarray:
        """Impute lactate / albumin values given the provided features. Don't
            need to winsorize here as transformer is fit to winsorized data. If
            probabilitic is True, the imputed value for each patient is a
            single sample from the patient-specific distribution over lactate
            or albumin. If probabilitic is False, the imputed value is the
            mean of that distribution (note that lac_alb_imp_i is ignored and
            must be None in this case)."""
        if probabilistic:
            assert isinstance(lac_alb_imp_i, int)
            lacalb_imputed_trans = quick_sample(
                gam=self.imputers[split_i],
                sample_at_X=features.values,
                quantity='y',
                n_draws=1,
                random_seed=lac_alb_imp_i
            ).flatten()
        else:
            assert lac_alb_imp_i is None
            lacalb_imputed_trans = self.imputers[split_i].predict_mu(
                X=features.values)
        return self.transformers[split_i].inverse_transform(
            lacalb_imputed_trans.reshape(-1, 1))

    def get_complete_variable_and_missingness_indicator(
        self,
        fold_name: str,
        split_i: int,
        mice_imp_i: int,
        lac_alb_imp_i: int,
        missingness_indicator: bool = True
    ) -> pd.DataFrame:
        """Impute missing albumin / lactate values, then use the observed and
            imputed values to construct a DataFrame with a single complete
            column of albumin / lactate values. Optionally, add a second column
            which is 1 where values were originally missing, otherwise 0."""
        missing_features = self._get_features_where_lacalb_missing(
            fold_name, split_i, mice_imp_i)
        lacalb_imputed = self.impute(
            missing_features, split_i, lac_alb_imp_i, probabilistic=True)
        lacalb = self._get_complete_lacalb(
            lacalb_imputed, fold_name, split_i)
        lacalb = self._winsorize(split_i, lacalb)
        if missingness_indicator:
            lacalb = self._add_missingness_indicator(lacalb, fold_name, split_i)
        return lacalb

    def get_observed_and_predicted(
        self,
        fold_name: str,
        split_i: int,
        mice_imp_i: int,
        lac_alb_imp_i: Union[int, None],
        probabilistic: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience function which fetches the observed lactate / albumin
            values from a given fold in a given train-test split, and the
            corresponding lactate / albumin values predicted by the imputation
            model."""
        if fold_name == 'train':
            lacalb, _, _, _ = self._split(split_i)
        elif fold_name == 'test':
            _, _, lacalb, _ = self._split(split_i)
        obs_lacalb = self._get_observed_values(
            fold=fold_name,
            split_i=split_i,
            X=lacalb
        )
        obs_lacalb = self._winsorize(split_i, obs_lacalb)
        features = self._get_features_where_lacalb_observed(
            fold_name, split_i, mice_imp_i)
        pred_lacalb = self.impute(
            features,
            split_i,
            lac_alb_imp_i,
            probabilistic=probabilistic
        ).flatten()
        return obs_lacalb[self.lacalb_variable_name].values, pred_lacalb

    def get_all_observed_and_predicted(
        self,
        fold_name: str,
        lac_alb_imp_i: Union[int, None],
        probabilistic: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Convenience function which fetches the observed lactate / albumin
            values from a given fold in EVERY given train-test split, and the
            corresponding lactate / albumin values predicted by the imputation
            model."""
        y_obs, y_preds = [], []
        for split_i in pb(range(self.tts.n_splits), prefix="Split iteration"):
            for mice_imp_i in range(self.cat_imputer.swm.n_mice_imputations):
                y_ob, y_pred = self.get_observed_and_predicted(
                    fold_name=fold_name,
                    split_i=split_i,
                    mice_imp_i=mice_imp_i,
                    lac_alb_imp_i=lac_alb_imp_i,
                    probabilistic=probabilistic
                )
                y_obs.append(y_ob)
                y_preds.append(y_pred)
        return y_obs, y_preds

    def _get_features_where_lacalb_missing(
        self,
        fold_name: str,
        split_i: int,
        mice_imp_i: int
    ) -> pd.DataFrame:
        features = self.cat_imputer.get_imputed_df(
            fold_name, split_i, mice_imp_i)
        if not self.mortality_as_feature:
            features = features.drop(self.target_variable_name, axis=1)
        return features.loc[
            self.missing_i[fold_name][split_i][self.lacalb_variable_name]]

    def _get_features_where_lacalb_observed(
        self,
        fold_name: str,
        split_i: int,
        mice_imp_i: int
    ) -> pd.DataFrame:
        features = self.cat_imputer.get_imputed_df(
            fold_name, split_i, mice_imp_i)
        if not self.mortality_as_feature:
            features = features.drop(self.target_variable_name, axis=1)
        return features.loc[features.index.difference(
            self.missing_i[fold_name][split_i][self.lacalb_variable_name])]

    def _get_complete_lacalb(
        self,
        lacalb_imputed: np.ndarray,
        fold_name: str,
        split_i: int
    ) -> pd.DataFrame:
        if fold_name == 'train':
            lacalb, _, _, _ = self._split(split_i)
        elif fold_name == 'test':
            _, _, lacalb, _ = self._split(split_i)
        lacalb.loc[
            self.missing_i[fold_name][split_i][self.lacalb_variable_name]
        ] = lacalb_imputed
        return lacalb

    def _add_missingness_indicator(
        self,
        df: pd.DataFrame,
        fold_name: str,
        split_i: int
    ) -> pd.DataFrame:
        """Adds a missingness indicator column for imputed variable."""
        missing_i_name = f"{self.lacalb_variable_name}_missing"
        df[missing_i_name] = np.zeros(df.shape[0])
        df.loc[
            self.missing_i[fold_name][split_i][self.lacalb_variable_name],
            missing_i_name
        ] = 1.0
        return df

    def get_imputed_variables(self, fold_name, split_i, imp_i):
        """Override base class method which shouldn't be used, as we don't
            store the imputed lactate / albumin values, but rather store the
            imputation models and use them to impute as and when needed."""
        raise NotImplementedError


class NovelModel:
    """Handles the process of repeated of train-test splitting, re-fitting the
        novel mortality model using the training fold. Also allows prediction
        of mortality risk distribution for each case in the test fold."""

    def __init__(
        self,
        categorical_imputer: CategoricalImputer,
        albumin_imputer: LactateAlbuminImputer,
        lactate_imputer: LactateAlbuminImputer,
        model_factory: Callable[
            [pd.Index, Dict[str, Tuple], str], LogisticGAM],
        n_lacalb_imputations_per_mice_imp: int,
        random_seed
    ):
        self.cat_imputer = categorical_imputer
        self.alb_imputer = albumin_imputer
        self.lac_imputer = lactate_imputer
        self.model_factory = model_factory
        self.n_lacalb_imp = n_lacalb_imputations_per_mice_imp
        self.random_seed = random_seed
        self.target_variable_name = categorical_imputer.swm.target_variable_name
        self.models: Dict[
            int,  # train-test split index
            LogisticGAM
        ] = {}

    def _calculate_lac_alb_imp_i(
        self,
        mice_imp_i: int,
        lac_alb_imp_i: int
    ) -> int:
        return self.random_seed + lac_alb_imp_i + self.n_lacalb_imp * mice_imp_i

    def get_features_and_labels(
        self,
        fold_name: str,
        split_i: int,
        mice_imp_i: int,
        lac_alb_imp_i: int
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Uses results of previous imputation to construct complete DataFrames
            of features, and corresponding mortality labels, ready for input to
            mortality risk prediction model."""
        df = self.cat_imputer.get_imputed_df(
            fold_name, split_i, mice_imp_i
        )
        target = df[self.target_variable_name].values
        features = df.drop(self.target_variable_name, axis=1)
        for imputer in (self.alb_imputer, self.lac_imputer):
            lacalb_and_indicator = (
                imputer.get_complete_variable_and_missingness_indicator(
                    fold_name=fold_name,
                    split_i=split_i,
                    mice_imp_i=mice_imp_i,
                    lac_alb_imp_i=self._calculate_lac_alb_imp_i(
                        mice_imp_i,
                        lac_alb_imp_i
                    )
                )
            )
            features = pd.concat([features, lacalb_and_indicator], axis=1)
        return features, target

    def fit(self):
        """Fit mortality risk models for every train-test split."""
        for split_i in pb(
            range(self.cat_imputer.tts.n_splits),
            prefix="Split iteration"
        ):
            self._single_train_test_split(split_i)

    def _single_train_test_split(self, split_i: int):
        """Fit combined mortality risk model for a single train-test split."""
        gams = []
        for mice_imp_i in range(self.cat_imputer.swm.n_mice_imputations):
            for lac_alb_imp_i in range(self.n_lacalb_imp):
                features, target = self.get_features_and_labels(
                    fold_name='train',
                    split_i=split_i,
                    mice_imp_i=mice_imp_i,
                    lac_alb_imp_i=self._calculate_lac_alb_imp_i(
                        mice_imp_i,
                        lac_alb_imp_i
                    )
                )
                gam = self.model_factory(
                    features.columns,
                    self.alb_imputer.multi_cat_vars,
                    self.lac_imputer.ind_var_name
                )
                gam.fit(features.values, target)
                gams.append(gam)
        self.models[split_i] = combine_mi_gams(gams)

    def get_observed_and_predicted(
        self,
        fold_name: str,
        split_i: int,
        n_samples_per_imp_i: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample predicted mortality risks for the train or test fold of a
        given train-test split. Also fetches the corresponding mortality
        labels (these are the same regardless of mice_imp_i and lac_alb_imp_i.

        Returns:
            Observed mortality labels in {0, 1}. Shape is (n_patients_in_fold,)
            Sampled predicted mortality risks in [0, 1]. Shape is
                (n_samples_per_patient, n_patients_in_fold,) where
                n_samples_per_patient = self.cat_imputer.swm.n_mice_imputations
                * self.n_lacalb_imp * n_samples_per_imp_i
        """
        mortality_risks = []
        for mice_imp_i in range(self.cat_imputer.swm.n_mice_imputations):
            for lac_alb_imp_i in range(self.n_lacalb_imp):
                features, labels = self.get_features_and_labels(
                    fold_name=fold_name,
                    split_i=split_i,
                    mice_imp_i=mice_imp_i,
                    lac_alb_imp_i=self._calculate_lac_alb_imp_i(
                        mice_imp_i,
                        lac_alb_imp_i
                    )
                )
                mortality_risks.append(
                    quick_sample(
                        gam=self.models[split_i],
                        sample_at_X=features.values,
                        quantity="mu",
                        n_draws=n_samples_per_imp_i,
                        random_seed=self._calculate_lac_alb_imp_i(
                            mice_imp_i,
                            lac_alb_imp_i
                        )
                    )
                )
        return labels, np.vstack(mortality_risks)

    def get_all_observed_and_median_predicted(
        self,
        fold_name: str,
        n_samples_per_imp_i: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Convenience function for preparing input to LogisticScorer. Fetches
        the observed mortality labels from a given fold in every train-test
        split, and the corresponding *median* mortality risks predicted by the
        novel model.

        Returns:
            Each element of this list is an ndarray of observed mortality
                labels in {0, 1}. Shape of each ndarray is (n_patients_in_fold,)
            Each element of this list is an ndarray of median mortality risks
                in [0, 1]. Shape of each ndarray is (n_patients_in_fold,)
        """
        y_obs, y_preds = [], []
        for split_i in pb(
            range(self.cat_imputer.tts.n_splits),
            prefix="Split iteration"
        ):
            y_ob, y_pred = self.get_observed_and_predicted(
                fold_name=fold_name,
                split_i=split_i,
                n_samples_per_imp_i=n_samples_per_imp_i
            )
            y_obs.append(y_ob)
            y_preds.append(np.median(y_pred, axis=0))
        return y_obs, y_preds
