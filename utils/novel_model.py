import operator
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

NOVEL_MODEL_VARS = {
    'cat': ('S03ASAScore',
            'S03CardiacSigns',
            'S03RespiratorySigns',
            'S03DiagnosedMalignancy',
            'S03Pred_Peritsoil',
            'S02PreOpCTPerformed',
            'S03ECG'),
    'cont': ('S01AgeOnArrival',
             'S03SerumCreatinine',
             'S03PreOpArterialBloodLactate',
             'S03PreOpLowestAlbumin',
             'S03Sodium',
             'S03Potassium',
             'S03Urea',
             'S03WhiteCellCount',
             'S03Pulse',
             'S03SystolicBloodPressure',
             'S03GlasgowComaScore'),
    # plus Indications variable
    'target': 'Target'}

# Need to add levels for the Indications variable below, once they are derived
MULTI_CATEGORY_LEVELS = {
    'S03ASAScore': (1., 2., 3., 4., 5.),
    'S03CardiacSigns': (1., 2., 4., 8.),
    'S03RespiratorySigns': (1., 2., 4., 8.),
    'S03DiagnosedMalignancy': (1., 2., 4., 8.),
    'S03Pred_Peritsoil': (1., 2., 4., 8.)
}

MISSINGNESS_VARS = ('S03PreOpArterialBloodLactate', 'S03PreOpLowestAlbumin')
MISSINGNESS_SUFFIX = '_missing'


def combine_categories(
        df: pd.DataFrame,
        category_mapping: Dict[str, Dict[float, float]]
) -> pd.DataFrame:
    """Combines values of categorical variables. Propogates missing values.
        each key-value pair in combine specifies a remappng of current
        categories to new ones. An example key-value pair in combine is
        'S03ECG' : {1.0: 0.0, 4.0: 1.0, 8.0: 1.0} which combines
        the two 'abnormal ecg' categories (4.0 and 8.0) together."""
    drop = []

    for v, mapping in category_mapping.items():
        temp_variable_name = f'{v}_temp'
        drop.append(temp_variable_name)
        df[temp_variable_name] = df[v].copy()
        df[v] = np.nan

        for old, new in mapping.items():
            df.loc[df[temp_variable_name] == old, v] = new

    return df.drop(drop, axis=1)


def add_missingness_indicators(df: pd.DataFrame,
                               variables: Tuple[str]) -> pd.DataFrame:
    """Adds a missingness indicator column for each of the specified
        variables."""
    for v in variables:
        c_missing = f'{v}_missing'
        df[c_missing] = np.zeros(df.shape[0])
        df.loc[df[v].isnull(), c_missing] = 1.
    return df


def ohe_to_single_column(df: pd.DataFrame,
                         variable_name: str,
                         categories: List[str]) -> pd.DataFrame:
    """Changes a variable that is one-hot encoded over multiple DataFrame
        columns to integers in a single column."""
    df[variable_name] = df[categories].idxmax(axis=1)
    return df.drop(categories, axis=1)


def label_encode(
        df: pd.DataFrame,
        multi_cat_levels: Dict,
        missing_indication_value: str) -> pd.DataFrame:
    """Encode labels for each novel-model categorical variable as integers, with
        missingness support."""
    for c, levels in multi_cat_levels.items():
        if c is not 'Indication':
            df[c] = df[c].astype(float)
            df[c] = [np.nan if np.isnan(x) else levels.index(x)
                     for x in df[c].values]
            df[c] = df[c].astype(float)
        else:
            df[c] = [np.nan if x == missing_indication_value
                     else levels.index(x) for x in df[c].values]
    return df


def winsorize_novel(
        df: pd.DataFrame,
        thresholds_dict: Dict[str, Tuple[float, float]] = None,
        quantiles: Tuple[float, float] = (0.001, 0.999),
        cont_vars: List[str] = None,
        include: Dict[str, Tuple[bool, bool]] = None
) -> (pd.DataFrame, Dict[str, Tuple[float, float]]):
    """Winsorize continuous variables at thresholds in thresholds_dict, or at
        specified quantiles if thresholds_dict is None. If thresholds_dict is
        None, upper and/or lower Winsorization for selected variables can be
        disabled using the include dict. Variables not specified in the include
        dict have Winsorization applied at upper and lower thresholds by
        default."""
    df = df.copy()
    ops = (operator.lt, operator.gt)

    if thresholds_dict:
        for v, thresholds in thresholds_dict.items():
            for i, threshold in enumerate(thresholds):
                if threshold is not None:
                    df.loc[ops[i](df[v], threshold), v] = threshold
    else:
        thresholds_dict = {}
        for v in cont_vars:
            thresholds_dict[v] = list(df[v].quantile(quantiles))
            for i, threshold in enumerate(thresholds_dict[v]):
                try:
                    if include[v][i]:
                        df.loc[ops[i](df[v], threshold), v] = threshold
                    else:
                        thresholds_dict[v][i] = None
                except KeyError:
                    df.loc[ops[i](df[v], threshold), v] = threshold

    return df, thresholds_dict


def preprocess_novel_pre_split(
        df: pd.DataFrame,
        category_mapping: Dict[str, Dict[float, float]],
        missingness_indicator_variables: List[str],
        indication_variable_name: str,
        indications: List[str],
        missing_indication_value: str,
        multi_category_levels: Dict
) -> pd.DataFrame:
    """In preparation for later data input to the novel model, does the data
        preprocessing steps which can be safely performed before train-test
        splitting."""
    df = df.copy()
    df = combine_categories(df, category_mapping)
    df = add_missingness_indicators(df, missingness_indicator_variables)
    df = ohe_to_single_column(df, indication_variable_name, indications)
    df = label_encode(df, multi_category_levels, missing_indication_value)
    return df
