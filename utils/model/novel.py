import operator
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
import pandas as pd
from progressbar import progressbar as pb
from pygam import LogisticGAM, s, f, te

from utils.gam import combine_mi_gams, quick_sample
from utils.impute import CategoricalImputer, LactateAlbuminImputer
from utils.indications import ohe_to_single_column

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
    # TODO: Should GCS splines have lower order?
    # TODO: Consider edge knots
    # TODO: Consider reducing n_splines for most continuous variables
    # TODO: Indications should be more regularised?
    return LogisticGAM(
        s(columns.get_loc("S01AgeOnArrival"), lam=200)
        + s(columns.get_loc("S03SystolicBloodPressure"), lam=300)
        + te(
            columns.get_loc("S03Pulse"),
            columns.get_loc("S03ECG"),
            lam=(250, 2),
            n_splines=(20, 2),
            spline_order=(3, 0),
            dtype=("numerical", "categorical"),
        )
        + s(columns.get_loc("S03WhiteCellCount"), lam=50)
        + s(columns.get_loc("S03Sodium"), lam=220)
        + s(columns.get_loc("S03Potassium"), lam=300)
        + s(columns.get_loc(LACTATE_VAR_NAME), lam=150)
        + s(columns.get_loc(ALBUMIN_VAR_NAME), lam=150)
        + s(columns.get_loc("S03GlasgowComaScore"), n_splines=13, lam=150)
        + f(columns.get_loc("S03ASAScore"), coding="dummy", lam=50)
        + f(
            columns.get_loc(f"{LACTATE_VAR_NAME}{MISSINGNESS_SUFFIX}"),
            coding="dummy",
            lam=200
        )
        + f(
            columns.get_loc(f"{ALBUMIN_VAR_NAME}{MISSINGNESS_SUFFIX}"),
            coding="dummy",
            lam=200
        )
        + te(
            columns.get_loc("S03DiagnosedMalignancy"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(200, 200),
            n_splines=(len(multi_cat_levels["S03DiagnosedMalignancy"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03Pred_Peritsoil"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(400, 200),
            n_splines=(len(multi_cat_levels["S03Pred_Peritsoil"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03CardiacSigns"),
            columns.get_loc("S03RespiratorySigns"),
            lam=150,
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
            lam=18.0,
            dtype=("numerical", "numerical"),
        )
        + te(
            columns.get_loc(indication_var_name),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(30, 200),
            n_splines=(len(multi_cat_levels[indication_var_name]), 2),
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


def add_missingness_indicators(df: pd.DataFrame,
                               variables: List[str]) -> pd.DataFrame:
    """Adds a missingness indicator column for each of the specified
        variables."""
    for v in variables:
        c_missing = f"{v}_missing"
        df[c_missing] = np.zeros(df.shape[0])
        df.loc[df[v].isnull(), c_missing] = 1.0
    return df


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


class NovelModel:
    """Handles the process of repeated of train-test splitting, re-fitting the
        novel mortality model using the training fold. Also allows prediction of
        mortality risk distribution for each case in the test fold."""

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
                imputer.get_imputed_variable_and_missingness_indicator(
                    fold_name,
                    split_i,
                    mice_imp_i,
                    self._calculate_lac_alb_imp_i(mice_imp_i, lac_alb_imp_i)
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

    def predict(
        self,
        fold_name: str,
        split_i: int,
        n_samples_per_imp_i: int
    ) -> np.ndarray:
        """Sample predicted mortality risks for the train or test fold of a 
            given train-test split."""
        mortality_risks = []
        for mice_imp_i in range(self.cat_imputer.swm.n_mice_imputations):
            for lac_alb_imp_i in range(self.n_lacalb_imp):
                features, _ = self.get_features_and_labels(
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
        return np.vstack(mortality_risks)
