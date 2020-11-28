import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from progressbar import progressbar as pb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from pyarrow import feather

from utils.io import make_directory
from utils.split import Splitter, TrainTestSplitter

"""
Web form for current NELA risk model includes Hb, but only uses this to
    calculate P-POSSUM.

Winsorization thresholds and centres are sourced from the appendix of
    Eugene et al. BJA 2018. Available at https://bit.ly/3dE9YS2
"""

CURRENT_MODEL_VARS = {
    "cat": (
        "S01Sex",
        "S03ASAScore",
        "S03NCEPODUrgency",
        "S03ECG",
        "S03NumberOfOperativeProcedures",
        "S03CardiacSigns",
        "S03RespiratorySigns",
        "S03Pred_Peritsoil",
        "S03Pred_TBL",
        "S03DiagnosedMalignancy",
        "S03WhatIsTheOperativeSeverity",
        "S03GlasgowComaScore",
    ),
    "cont": (
        "S01AgeOnArrival",
        "S03SerumCreatinine",
        "S03Sodium",
        "S03Potassium",
        "S03Urea",
        "S03WhiteCellCount",
        "S03Pulse",
        "S03SystolicBloodPressure",
    ),
    "target": "Target",
}

WINSOR_THRESHOLDS = {
    # age seems to be excluded from winsorization
    "logcreat": (3.3, 6.0),
    "S03SystolicBloodPressure": (70.0, 190.0),
    "S03Pulse": (55.0, 145.0),
    "S03WhiteCellCount": (1.0, 42.7),
    "logurea": (0.0, 3.7),
    "S03Potassium": (2.8, 5.9),
    "S03Sodium": (124.0, 148.0),
}

CENTRES = {
    "S01AgeOnArrival": -64.0,
    "logcreat": -4.0,
    "S03SystolicBloodPressure": -127.0,
    "S03Pulse": -91.0,
    "S03WhiteCellCount": -13.0,
    "logurea": -1.9,
    "S03Potassium": -4.0,
    "S03Sodium": -123.0,
}

CONTINUOUS_VARIABLES_AFTER_PREPROCESSING = (
    'S03Potassium',
    'S03WhiteCellCount',
    'S03Pulse',
    'S03SystolicBloodPressure',
    'logcreat',
    'logurea',
    'S03Potassium_2',
    'S03WhiteCellCount_2',
    'S03Pulse_2',
    'S03SystolicBloodPressure_2',
    'logcreat_2',
    'logurea_2',
    'S03Sodium_3',
    'S03Sodium_3_log',
    'age_asa12',
    'age_2_asa12',
    'age_asa3',
    'age_2_asa3',
    'age_asa4',
    'age_2_asa4',
    'age_asa5',
    'age_2_asa5'
)


def discretise_gcs(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise GCS. 13-15 category is eliminated to avoid dummy variable
        effect."""
    for gcs_threshold in ((3, 9), (9, 13)):
        v_name = f"gcs_{gcs_threshold[0]}_{int(gcs_threshold[1] - 1)}"
        df[v_name] = np.zeros(df.shape[0])
        df.loc[
            (df["S03GlasgowComaScore"] >= gcs_threshold[0])
            & (df["S03GlasgowComaScore"] < gcs_threshold[1]),
            v_name,
        ] = 1
    return df.drop("S03GlasgowComaScore", axis=1)


def combine_ncepod_urgencies(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NCEPOD urgency 4 (Emergency but resuscitation of >2h
        possible) to category 3 (Urgent 2-6h) as the current model doesn't
        include category 4."""
    df.loc[df["S03NCEPODUrgency"] == 4.0, "S03NCEPODUrgency"] = 3.0
    return df


def combine_highest_resp_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Combine 2 highest levels of respiratory disease."""
    df.loc[df["S03RespiratorySigns"] == 8.0, "S03RespiratorySigns"] = 4.0
    return df


def binarize_categorical(
    df: pd.DataFrame,
    label_binarizers: Union[None, List[LabelBinarizer]],
    binarize_vars: List[str],
) -> (pd.DataFrame, List[LabelBinarizer]):
    """Binarize categorical features. Can use pre-fit label binarizers
        if these are available (e.g. ones fit on the train fold).

        Note that in the current NELA model, GCS is exempt from binarization
        as it is binned. ASA is exempt as it is only used in a later interaction
        term."""
    if label_binarizers:
        lb = label_binarizers
    else:
        lb = {}

    for v in binarize_vars:
        if label_binarizers:
            trans = lb[v].transform(df[v].astype(int).values)
        else:
            lb[v] = LabelBinarizer()
            trans = lb[v].fit_transform(df[v].astype(int).values)

        if len(lb[v].classes_) == 2:
            df[v] = trans
        else:
            cat_names = [f"{v}_{c}" for c in lb[v].classes_]
            v_df = pd.DataFrame(
                data=trans,
                columns=cat_names,
                index=df.index
            )
            df = pd.concat([df, v_df], axis=1)
            df = df.drop(v, axis=1)

    return df, label_binarizers


def drop_base_categories(df: pd.DataFrame) -> pd.DataFrame:
    """For each categorical variable with k categories, we need to
        end up with k-1 variables to avoid the dummy variable trap.
        So, we drop the category that corresponds to baseline risk in
        for each variable."""
    return df.drop([
        "S03RespiratorySigns_1",
        "S03NCEPODUrgency_1",
        "S03ECG_1",
        "S03NumberOfOperativeProcedures_1",
        "S03CardiacSigns_1",
        "S03Pred_Peritsoil_1",
        "S03Pred_TBL_1",
        "S03DiagnosedMalignancy_1",
    ], axis=1)


def log_urea_creat(df: pd.DataFrame) -> pd.DataFrame:
    """Log transform urea and creatinine."""
    df["logcreat"] = np.log(df["S03SerumCreatinine"])
    df["logurea"] = np.log(df["S03Urea"])
    return df.drop(["S03SerumCreatinine", "S03Urea"], axis=1)


def winsorize_current(
    df: pd.DataFrame, winsor_thresholds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """Winsorize continuous variables at thresholds specified in NELA paper
        appendix."""
    for v, threshold in winsor_thresholds.items():
        df.loc[df[v] < threshold[0], v] = threshold[0]
        df.loc[df[v] > threshold[1], v] = threshold[1]
    return df


def centre(df: pd.DataFrame, centres: Dict[str, float]) -> pd.DataFrame:
    """Centre continuous variables at centres specified in NELA paper
        appendix."""
    for v, c in centres.items():
        df.loc[:, v] += c
    return df


def add_quadratic_features(df: pd.DataFrame,
                           quadratic_vars: List[str]) -> pd.DataFrame:
    """Add quadratic transformation of some continuous features. Sodium is
        exempt as it undergoes a customised transformation."""
    for v in quadratic_vars:
        df[f"{v}_2"] = df[v] ** 2
    return df


def add_asa_age_resp_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Make separate age, age^2 and respiratory status features for levels of
        ASA. ASA 1 and 2 are grouped together."""
    for (interaction, v) in (
        ("age_asa12", "S01AgeOnArrival"),
        ("age_2_asa12", "S01AgeOnArrival_2"),
        ("resp2_asa12", "S03RespiratorySigns_2"),
        ("resp4_asa12", "S03RespiratorySigns_4"),
    ):
        df[interaction] = np.zeros(df.shape[0])
        df.loc[
            (df["S03ASAScore"] == 1) | (df["S03ASAScore"] == 2), interaction
        ] = df.loc[(df["S03ASAScore"] == 1) | (df["S03ASAScore"] == 2), v]

    for asa in range(3, 6):
        for (interaction, v) in (
            (f"age_asa{asa}", "S01AgeOnArrival"),
            (f"age_2_asa{asa}", "S01AgeOnArrival_2"),
            (f"resp2_asa{asa}", "S03RespiratorySigns_2"),
            (f"resp4_asa{asa}", "S03RespiratorySigns_4"),
        ):
            df[interaction] = np.zeros(df.shape[0])
            df.loc[df["S03ASAScore"] == asa, interaction] = df.loc[
                df["S03ASAScore"] == asa, v
            ]

    return df.drop(
        [
            "S01AgeOnArrival",
            "S01AgeOnArrival_2",
            "S03RespiratorySigns_2",
            "S03RespiratorySigns_4",
            "S03ASAScore",
        ],
        axis=1,
    )


def transform_sodium(df: pd.DataFrame) -> pd.DataFrame:
    """Transform sodium as described in NELA paper."""
    df["S03Sodium_3"] = df["S03Sodium"] ** 3
    df["S03Sodium_3_log"] = df["S03Sodium_3"] * np.log(df["S03Sodium"])
    return df.drop("S03Sodium", axis=1)


def preprocess_current(
    df: pd.DataFrame,
    quadratic_vars: List[str],
    winsor_thresholds: Dict[str, Tuple[float, float]],
    centres: Dict[str, float],
    binarize_vars: List[str],
    label_binarizers: Union[None, List[LabelBinarizer]],
) -> (pd.DataFrame, List[LabelBinarizer]):
    """Preprocess NELA data for input to current EL mortality risk model.

    Args:
        df: Manually-wrangled NELA data with some preprocessing, and incomplete
            cases removed (the DataFrame index should not be reset until
            after train-test splitting)
        quadratic_vars: Continuous features to add a quadratic transformation of
        winsor_thresholds: Upper and lower bounds for winsorization of
            continuous variables
        centres: For use in centering the continuous variables
        binarize_vars: Categorical features to binarize
        label_binarizers: The LabelBinarizer objects used to binarize the
            categorical features. If None, LabelBinarizers will be fit using
            the data in df

    Returns:
        Preprocessed data
        Fitted LabelBinarizers (if label_binarizers is not None, then
            label_binarizers is returned unmodified
    """
    df = df.copy()

    # Preprocess categorical variables
    df = discretise_gcs(df)
    df = combine_ncepod_urgencies(df)
    df = combine_highest_resp_levels(df)
    df, label_binarizers = binarize_categorical(df, label_binarizers,
                                                binarize_vars)
    df = drop_base_categories(df)

    # Preprocess continuous variables
    df = log_urea_creat(df)
    df = winsorize_current(df, winsor_thresholds)
    df = centre(df, centres)
    df = add_quadratic_features(df, quadratic_vars)
    df = add_asa_age_resp_interaction(df)
    df = transform_sodium(df)

    return df, label_binarizers


class CurrentModel(Splitter):
    """Handles the process of repeated of train-test splitting, re-fitting the
        current NELA model using the training fold, and predicting mortality
        risk for each case in the test fold. Keeps track of relevant
        quantities."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_test_splitter: TrainTestSplitter,
        target_variable_name: str,
        random_seed,
    ):
        """
        Args:
            df: Preprocessed NELA data. When using with current model,
                incomplete cases should have been removed but the DataFrame
                index should not be reset
            train_test_splitter: TrainTestSplitter instance from
                01_train_test_split.py
            target_variable_name: Name of DataFrame column containing mortality
                status
            random_seed: Used to seed the sklearn LogisticRegression models
        """
        super().__init__(df, train_test_splitter, target_variable_name)
        self.features: List[str] = ["intercept"]
        self.coefficients: List[np.ndarray] = []
        self.y_test: List[np.ndarray] = []
        self.y_pred: List[np.ndarray] = []
        self.rnd = np.random.RandomState(random_seed)

    def split_train_predict(self) -> None:
        for i in pb(range(self.tts.n_splits), prefix="STP iteration"):
            X_train, y_train, X_test, y_test = self._split(i)
            model = self._train(X_train, y_train)
            self.y_test.append(y_test)
            self.y_pred.append(model.predict_proba(X_test.values)[:, 1])
        self.features += X_train.columns.tolist()

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray
    ) -> LogisticRegression:
        """We use the liblinear solver, as the unscaled features would slow the
            convergence of the other solvers. The current NELA model is
            unregularized, but using the liblinear solver means that we must
            specify a value for C (the inverse of regularisation strength). We
            get around this by setting C to a very large number in order to
            avoid any meaningful regularisation."""
        model = LogisticRegression(
            C=10 ** 50, solver="liblinear", random_state=self.rnd
        )
        model.fit(X_train.values, y_train)
        coefficients = model.coef_.flatten()
        self.coefficients.append(np.zeros(coefficients.shape[0] + 1))
        self.coefficients[-1][0] = model.intercept_
        self.coefficients[-1][1:] = coefficients
        return model


class CurrentModelDataExporter(Splitter):
    """Splits current model data, scales its continuous variables (otherwise
        R's glmer() doesn't converge), then exports as .feather for later
        random-effects logistic model fitting / prediction in R."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_test_splitter: TrainTestSplitter,
        target_variable_name: str,
        continuous_variables: Tuple[str, ...],
        save_parent_directory: str
    ):
        """
        Args:
            df: Preprocessed NELA data. When using with current model,
                incomplete cases should have been removed but the DataFrame
                index should not be reset
            train_test_splitter: TrainTestSplitter instance from
                01_train_test_split.py
            target_variable_name: Name of DataFrame column containing mortality
                status
            continuous_variables: Names of continuous variables in DataFrame
                output by preprocess_current
            save_parent_directory: Parent directory path for saving the .feather
                files. They will be saved in parent_directory/train and
                parent_directory/test
        """
        super().__init__(df, train_test_splitter, target_variable_name)
        self.cont_vars = continuous_variables
        self.save_dir = save_parent_directory

    def export(self) -> None:
        for i in pb(range(self.tts.n_splits), prefix="Split iteration"):
            X_train, y_train, X_test, y_test = self._split(i)
            folds = {
                'train': self._combine_X_y(X_train, y_train),
                'test': self._combine_X_y(X_test, y_test)
            }
            folds = self._rename_hospital_variable(folds)
            folds = self._scale_continuous_variables(folds)
            self._save_feathers(folds, i)

    def _combine_X_y(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        X[self.target_variable_name] = y
        return X

    @staticmethod
    def _rename_hospital_variable(
        folds: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        for fold_name in folds.keys():
            folds[fold_name] = folds[fold_name].rename(
                {'HospitalId.anon': 'HospitalId'}, axis=1)
        return folds

    def _scale_continuous_variables(
        self,
        folds: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        scaler = StandardScaler()
        folds['train'].loc[:, self.cont_vars] = scaler.fit_transform(
            folds['train'].loc[:, self.cont_vars])
        folds['test'].loc[:, self.cont_vars] = scaler.transform(
            folds['test'].loc[:, self.cont_vars])
        return folds

    def _save_feathers(self, folds: Dict[str, pd.DataFrame], i: int) -> None:
        for fold_name in folds.keys():
            feather.write_feather(
                df=folds[fold_name],
                dest=os.path.join(
                    self.save_dir, fold_name, f'{i}_{fold_name}.feather'),
                compression='uncompressed')
