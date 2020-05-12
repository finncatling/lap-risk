from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


"""NB. Web form for current NELA risk model includes Hb, but only uses this to
calculate P-POSSUM."""
CURRENT_NELA_MODEL_VARS = {
    'cat': ('S01Sex',
            'S03ASAScore',
            'S03NCEPODUrgency',
            'S03ECG',
            'S03NumberOfOperativeProcedures',
            'S03CardiacSigns',
            'S03RespiratorySigns',
            'S03Pred_Peritsoil',
            'S03Pred_TBL',
            'S03DiagnosedMalignancy',
            'S03WhatIsTheOperativeSeverity',
            'S03GlasgowComaScore'),
    'cont': ('S01AgeOnArrival',
             'S03SerumCreatinine',
             'S03Sodium',
             'S03Potassium',
             'S03Urea',
             'S03WhiteCellCount',
             'S03Pulse',
             'S03SystolicBloodPressure'),
    'target': 'Target'}


WINSOR_THRESHOLDS = {
    # age seems to be excluded from winsorization
    'logcreat': (3.3, 6.0),
    'S03SystolicBloodPressure': (70.0, 190.0),
    'S03Pulse': (55.0, 145.0),
    'S03WhiteCellCount': (1.0, 42.7),
    'logurea': (0.0, 3.7),
    'S03Potassium': (2.8, 5.9),
    'S03Sodium': (124.0, 148.0)}


CENTRES = {
    'S01AgeOnArrival': -64.0,
    'logcreat': -4.0,
    'S03SystolicBloodPressure': -127.0,
    'S03Pulse': -91.0,
    'S03WhiteCellCount': -13.0,
    'logurea': -1.9,
    'S03Potassium': -4.0,
    'S03Sodium': -123.0}


def flatten_nela_var_dict(nela_vars: Dict) -> List[str]:
    """Flattens current NELA model variable name dict into single list."""
    return (list(nela_vars['cat']) + list(nela_vars['cont']) +
            [nela_vars['target']])


def discretise_gcs(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise GCS. 13-15 category is eliminated to avoid dummy variable
        effect."""
    for gcs_threshold in ((3, 9), (9, 13)):
        v_name = f'gcs_{gcs_threshold[0]}_{gcs_threshold[1]}'
        df[v_name] = np.zeros(df.shape[0])
        df.loc[(df['S03GlasgowComaScore'] >= gcs_threshold[0]) &
               (df['S03GlasgowComaScore'] < gcs_threshold[1]), v_name] = 1
    return df.drop('S03GlasgowComaScore', axis=1)


def combine_ncepod_urgencies(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NCEPOD urgency 4 (Emergency but resuscitation of >2h
        possible) to category 3 (Urgent 2-6h) as the current model doesn't
        include category 4."""
    df.loc[df['S03NCEPODUrgency'] == 4., 'S03NCEPODUrgency'] = 3.
    return df


def combine_highest_resp_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Combine 2 highest levels of respiratory disease."""
    df.loc[df['S03RespiratorySigns'] == 8., 'S03RespiratorySigns'] = 4.
    return df


def binarize_categorical(
    df: pd.DataFrame,
    label_binarizers: List[LabelBinarizer],
    cat_vars: List[str]
) -> (pd.DataFrame, List[LabelBinarizer]):
    """Binarise categorical features. Can use pre-fit label binarisers
        if these are available (e.g. ones fit on the train fold).

        GCS is exempt from binarisation as it is discretised differently in a
        separate function. ASA is exempt as it is only used in a later
        interaction term."""
    if label_binarizers:
        lb = label_binarizers
    else:
        lb = {}

    for v in cat_vars:
        if v not in ('S03GlasgowComaScore', 'S03ASAScore'):
            if label_binarizers:
                trans = lb[v].transform(df[v].astype(int).values)
            else:
                lb[v] = LabelBinarizer()
                trans = lb[v].fit_transform(df[v].astype(int).values)

            if len(lb[v].classes_) == 2:
                df[v] = trans
            else:
                cat_names = [f'{v}_{c}' for c in lb[v].classes_]
                v_df = pd.DataFrame(trans, columns=cat_names)
                df = pd.concat([df, v_df], axis=1)

    return df, label_binarizers


def drop_base_categories(df: pd.DataFrame) -> pd.DataFrame:
    """For each categorical variable with k categories, we need to
        end up with k-1 variables to avoid the dummy variable trap.
        So, we drop the category that corresponds to baseline risk in
        for each variable."""
    return df.drop(['S03RespiratorySigns_1',
                    'S03NCEPODUrgency_1',
                    'S03ECG_1',
                    'S03NumberOfOperativeProcedures_1',
                    'S03CardiacSigns_1',
                    'S03Pred_Peritsoil_1',
                    'S03Pred_TBL_1',
                    'S03DiagnosedMalignancy_1'], axis=1)


def log_urea_creat(df: pd.DataFrame) -> pd.DataFrame:
    """Log transform urea and creatinine."""
    df['logcreat'] = np.log(df['S03SerumCreatinine'])
    df['logurea'] = np.log(df['S03Urea'])
    return df.drop(['S03SerumCreatinine', 'S03Urea'], axis=1)


def winsorize(
    df: pd.DataFrame,
    winsor_thresholds: Dict[str, Tuple[float, float]]
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
                           cont_vars: List[str]) -> pd.DataFrame:
    """Add quadratic transformation of all continuous features."""
    # TODO: ensure that cont_log_vars (consider rename) is input to this
    for v in cont_vars:
        df[f'{v}_2'] = df[v] ** 2
    return df


def add_asa_age_resp_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Make separate age, age^2 and respiratory status features for levels of
        ASA. ASA 1 and 2 are grouped together."""
    for (interaction, v) in (('age_asa12', 'S01AgeOnArrival'),
                             ('age_2_asa12', 'S01AgeOnArrival_2'),
                             ('resp2_asa12', 'S03RespiratorySigns_2'),
                             ('resp4_asa12', 'S03RespiratorySigns_4')):
        df[interaction] = np.zeros(df.shape[0])
        df.loc[(df['S03ASAScore'] == 1) |
               (df['S03ASAScore'] == 2),
               interaction] = df.loc[(df['S03ASAScore'] == 1) |
                                     (df['S03ASAScore'] == 2),
                                     v]

    for asa in range(3, 6):
        for (interaction, v) in ((f'age_asa{asa}', 'S01AgeOnArrival'),
                                 (f'age_2_asa{asa}', 'S01AgeOnArrival_2'),
                                 (f'resp2_asa{asa}', 'S03RespiratorySigns_2'),
                                 (f'resp4_asa{asa}', 'S03RespiratorySigns_4')):
            df[interaction] = np.zeros(df.shape[0])
            df.loc[df['S03ASAScore'] == asa, interaction] = df.loc[
                df['S03ASAScore'] == asa, v]

    return df.drop(['S01AgeOnArrival', 'S01AgeOnArrival_2',
                    'S03RespiratorySigns_2', 'S03RespiratorySigns_4',
                    'S03ASAScore'], axis=1)


def transform_sodium(df: pd.DataFrame) -> pd.DataFrame:
    """Transform sodium as described in NELA paper."""
    # TODO: Remove sodium**2 (currently left in by accident)
    df['S03Sodium_3'] = df['S03Sodium'] ** 3
    df['S03Sodium_3_log'] = df['S03Sodium_3'] * np.log(df['S03Sodium'])
    return df.drop('S03Sodium', axis=1)


def preprocess_categorical(
    df: pd.DataFrame,
    label_binarizers: List[LabelBinarizer]
) -> (pd.DataFrame, List[LabelBinarizer]):
    """Preprocess categorical variables in NELA data."""
    df = discretise_gcs(df)
    df = combine_ncepod_urgencies(df)
    df = combine_highest_resp_levels(df)
    df, label_binarizers = binarize_categorical(df, label_binarizers)
    df = drop_base_categories(df)
    return df, label_binarizers


def preprocess_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess continuous variables in NELA data."""
    df = log_urea_creat(df)
    df = winsorize(df)
    df = centre(df)
    df = add_quadratic_features(df)
    df = add_asa_age_resp_interaction(df)
    df = transform_sodium(df)
    return df


def preprocess_df(data, label_binarizers=None):
    """Preprocess NELA data."""
    df = data.copy()
    df = df.reset_index(drop=True)
    df, label_binarizers = preprocess_categorical(df, label_binarizers)
    df = preprocess_continuous(df)
    return df, label_binarizers
