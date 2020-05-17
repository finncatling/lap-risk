from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.constants import MISSING_IND_CATEGORY

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
    'S03Pred_Peritsoil': (1., 2., 4., 8.),
    'S03NCEPODUrgency': (1., 2., 3., 8.)
}


def combine_categories(df: pd.DataFrame,
                       combine: Dict[str, Dict[float, float]]) -> pd.DataFrame:
    """Combines values of categorical variables. Propogates missing values.
        each key-value pair in combine specifies a remappng of current
        categories to new ones. An example key-value pair in combine is
        'S03ECG' : {1.0: 0.0, 4.0: 1.0, 8.0: 1.0} which combines
        the two 'abnormal ecg' categories (4.0 and 8.0) together."""
    drop = []

    for v, mapping in combine.items():
        temp_variable_name = f'{v}_temp'
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
        c_missing = f'{v}_missing'
        df[c_missing] = np.zeros(df.shape[0])
        df.loc[df[v].isnull(), c_missing] = 1.
    return df


def ohe_to_single_column(df: pd.DataFrame,
                         variable_name: str,
                         categories: Tuple[str]) -> pd.DataFrame:
    """Changes a variable that is one-hot encoded over multiple DataFrame
        columns to integers in a single column."""
    df[variable_name] = df[categories].idxmax(axis=1)
    return df.drop(categories, axis=1)


def label_encode(
        df: pd.DataFrame,
        multi_cat_levels: Dict,
        missing_indication_value: str = MISSING_IND_CATEGORY) -> pd.DataFrame:
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
