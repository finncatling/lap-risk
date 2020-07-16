import operator
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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
LACTATE_ALBUMIN_VARS = (
    LACTATE_VAR_NAME,
    f"{LACTATE_VAR_NAME}{MISSINGNESS_SUFFIX}",
    ALBUMIN_VAR_NAME,
    f"{ALBUMIN_VAR_NAME}{MISSINGNESS_SUFFIX}",
)
INDICATION_VAR_NAME = "Indication"
INDICATION_PREFIX = "S05Ind_"
MISSING_IND_CATEGORY = f"{INDICATION_PREFIX}Missing"


def combine_categories(
    df: pd.DataFrame, category_mapping: Dict[str, Dict[float, float]]
) -> pd.DataFrame:
    """Combines values of categorical variables. Propogates missing values.
        each key-value pair in combine specifies a remappng of current
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


def add_missingness_indicators(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Adds a missingness indicator column for each of the specified
        variables."""
    for v in variables:
        c_missing = f"{v}_missing"
        df[c_missing] = np.zeros(df.shape[0])
        df.loc[df[v].isnull(), c_missing] = 1.0
    return df


def ohe_to_single_column(
    df: pd.DataFrame, variable_name: str, categories: List[str]
) -> pd.DataFrame:
    """Changes a variable that is one-hot encoded over multiple DataFrame
        columns to integers in a single column."""
    df[variable_name] = df[categories].idxmax(axis=1)
    return df.drop(categories, axis=1)


def label_encode(
    df: pd.DataFrame, multi_cat_levels: Dict, missing_indication_value: str
) -> pd.DataFrame:
    """Encode labels for each novel-model categorical variable as integers, with
        missingness support."""
    for c, levels in multi_cat_levels.items():
        if c is not "Indication":
            df[c] = df[c].astype(float)
            df[c] = [np.nan if np.isnan(x) else levels.index(x) for x in df[c].values]
            df[c] = df[c].astype(float)
        else:
            df[c] = [
                np.nan if x == missing_indication_value else levels.index(x)
                for x in df[c].values
            ]
    return df


def winsorize_novel(
    df: pd.DataFrame,
    thresholds: Dict[str, Tuple[float, float]] = None,
    cont_vars: List[str] = None,
    quantiles: Tuple[float, float] = None,
    include: Dict[str, Tuple[bool, bool]] = None,
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
        for v, thresholds in thresholds.items():
            for i, threshold in enumerate(thresholds):
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
    add_missingness_indicators_for: List[str],
    indication_variable_name: str,
    indications: List[str],
    missing_indication_value: str,
    multi_category_levels: Dict,
) -> pd.DataFrame:
    """In preparation for later data input to the novel model, does the data
        preprocessing steps which can be safely performed before train-test
        splitting."""
    df = df.copy()
    df = combine_categories(df, category_mapping)
    df = add_missingness_indicators(df, add_missingness_indicators_for)
    df = ohe_to_single_column(df, indication_variable_name, indications)
    df = label_encode(df, multi_category_levels, missing_indication_value)
    return df
