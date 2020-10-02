import pandas as pd

from utils.model import current
from utils.model.current import (
    preprocess_current,
    WINSOR_THRESHOLDS,
    CURRENT_MODEL_VARS,
    CENTRES,
)
from utils.model.shared import flatten_model_var_dict
from utils.split import drop_incomplete_cases


def test_discretise_gcs():
    df = pd.DataFrame({"S03GlasgowComaScore": [3., 8., 9., 12., 13., 14]})
    df = current.discretise_gcs(df)
    assert pd.DataFrame({
        'gcs_3_8': [1., 1., 0., 0., 0., 0.],
        'gcs_9_12': [0., 0., 1., 1., 0., 0.]
    }).equals(df)


def test_binarize_categorical():
    df = pd.DataFrame({
        'binary': [5., 6., 5.],
        'multi_cat': [2., 3., 4.],
        'ignore': [1.5, 2.3, 9.1]
    }, index=[0, 2, 3])
    df, _ = current.binarize_categorical(
        df=df,
        label_binarizers=None,
        binarize_vars=['binary', 'multi_cat']
    )
    assert all(pd.DataFrame({
        'binary': [0, 1, 0],
        'ignore': [1.5, 2.3, 9.1],
        'multi_cat_2': [1, 0, 0],
        'multi_cat_3': [0, 1, 0],
        'multi_cat_4': [0, 0, 1]
    }, index=[0, 2, 3]) == df)


def test_winsorize_current():
    df = pd.DataFrame({
        'for_winsor': [6., 1., 20.],
        'ignore': [0., 1., 0.]
    })
    df = current.winsorize_current(
        df=df,
        winsor_thresholds={'for_winsor': (4., 10.)}
    )
    assert pd.DataFrame({
        'for_winsor': [6., 4., 10.],
        'ignore': [0., 1., 0.]
    }).equals(df)


def test_add_asa_age_resp_interaction():
    """Patient in row 0 fell into the (now dropped) base category 1.0 for
        respiratory signs, hence initial 0 in the S03RespiratorySigns_2 and
        03RespiratorySigns_4 columns."""
    df = pd.DataFrame({
        "S01AgeOnArrival": [1., 2., 3., 4., 5.],
        "S01AgeOnArrival_2": [1., 4., 9., 16., 25.],
        "S03RespiratorySigns_2": [0, 1, 1, 0, 0],
        "S03RespiratorySigns_4": [0, 0, 0, 1, 1],
        "S03ASAScore": [1., 2., 3., 4., 5.]
    })
    df = current.add_asa_age_resp_interaction(df)
    assert pd.DataFrame({
        "age_asa12": [1., 2., 0., 0., 0.],
        "age_2_asa12": [1., 4., 0., 0., 0.],
        "resp2_asa12": [0., 1., 0., 0., 0.],
        "resp4_asa12": [0., 0., 0., 0., 0.],
        "age_asa3": [0., 0., 3., 0., 0.],
        "age_2_asa3": [0., 0., 9., 0., 0.],
        "resp2_asa3": [0., 0., 1., 0., 0.],
        "resp4_asa3": [0., 0., 0., 0., 0.],
        "age_asa4": [0., 0., 0., 4., 0.],
        "age_2_asa4": [0., 0., 0., 16., 0.],
        "resp2_asa4": [0., 0., 0., 0., 0.],
        "resp4_asa4": [0., 0., 0., 1., 0.],
        "age_asa5": [0., 0., 0., 0., 5.],
        "age_2_asa5": [0., 0., 0., 0., 25.],
        "resp2_asa5": [0., 0., 0., 0., 0.],
        "resp4_asa5": [0., 0., 0., 0., 1.]
    }).equals(df)


def test_preprocess_current(initial_df_permutations_fixture):
    """End-to-end test which just checks that column names are as expected."""
    df = initial_df_permutations_fixture[
        flatten_model_var_dict(CURRENT_MODEL_VARS)]

    df, _ = drop_incomplete_cases(df)

    binarize_vars = list(CURRENT_MODEL_VARS["cat"])
    binarize_vars.remove("S03GlasgowComaScore")
    binarize_vars.remove("S03ASAScore")

    quadratic_vars = list(CURRENT_MODEL_VARS["cont"])
    quadratic_vars.remove("S03Sodium")
    for original, logged in (
        ("S03SerumCreatinine", "logcreat"),
        ("S03Urea", "logurea")
    ):
        quadratic_vars.remove(original)
        quadratic_vars.append(logged)

    preprocessed_df, _ = preprocess_current(
        df,
        quadratic_vars=quadratic_vars,
        winsor_thresholds=WINSOR_THRESHOLDS,
        centres=CENTRES,
        binarize_vars=binarize_vars,
        label_binarizers=None,
    )

    expected_column_names = {
        "S01Sex",
        "S03NCEPODUrgency_2",
        "S03NCEPODUrgency_3",
        "S03NCEPODUrgency_8",
        "S03ECG_4",
        "S03ECG_8",
        "S03NumberOfOperativeProcedures_4",
        "S03NumberOfOperativeProcedures_8",
        "S03CardiacSigns_2",
        "S03CardiacSigns_4",
        "S03CardiacSigns_8",
        "S03Pred_Peritsoil_2",
        "S03Pred_Peritsoil_4",
        "S03Pred_Peritsoil_8",
        "S03Pred_TBL_2",
        "S03Pred_TBL_4",
        "S03Pred_TBL_8",
        "S03DiagnosedMalignancy_2",
        "S03DiagnosedMalignancy_4",
        "S03DiagnosedMalignancy_8",
        "S03WhatIsTheOperativeSeverity",
        "gcs_3_8",
        "gcs_9_12",
        "logcreat",
        "logcreat_2",
        "logurea",
        "logurea_2",
        "S03Sodium_3",
        "S03Sodium_3_log",
        "S03Potassium",
        "S03Potassium_2",
        "S03WhiteCellCount",
        "S03WhiteCellCount_2",
        "S03Pulse",
        "S03Pulse_2",
        "S03SystolicBloodPressure",
        "S03SystolicBloodPressure_2",
        "age_asa12",
        "age_2_asa12",
        "resp2_asa12",
        "resp4_asa12",
        "age_asa3",
        "age_2_asa3",
        "resp2_asa3",
        "resp4_asa3",
        "age_asa4",
        "age_2_asa4",
        "resp2_asa4",
        "resp4_asa4",
        "age_asa5",
        "age_2_asa5",
        "resp2_asa5",
        "resp4_asa5",
        "Target"
    }

    assert set(preprocessed_df.columns) == expected_column_names
