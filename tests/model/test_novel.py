import copy
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from utils.indications import (
    INDICATION_VAR_NAME,
    MISSING_IND_CATEGORY,
    get_indication_variable_names
)
from utils.model import novel
from utils.model.novel import (
    NOVEL_MODEL_VARS,
    MULTI_CATEGORY_LEVELS,
    LACTATE_VAR_NAME,
    ALBUMIN_VAR_NAME,
    preprocess_novel_pre_split,
    winsorize_novel
)
from utils.model.shared import flatten_model_var_dict


@pytest.fixture()
def missing_categories_df_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        'a': [1., 2., 3., np.nan],
        'b': [4., 5., 4., 6.]
    })


def test_combine_categories(missing_categories_df_fixture):
    combined_df = novel.combine_categories(
        df=missing_categories_df_fixture,
        category_mapping={
            'a': {
                1.: 1.,
                2.: 2.,
                3.: 2.
            },
            'b': {
                4.: 6.,
                5.: 5.,
                6.: 6.
            },
        }
    )
    assert pd.DataFrame({
        'a': [1., 2., 2., np.nan],
        'b': [6., 5., 6., 6.]
    }).equals(combined_df)


def test_add_missingness_indicators(missing_categories_df_fixture):
    missing_indicator_df = novel.add_missingness_indicators(
        df=missing_categories_df_fixture,
        variables=['a']
    )
    assert pd.DataFrame({
        'a': [1., 2., 3., np.nan],
        'b': [4., 5., 4., 6.],
        'a_missing': [0., 0., 0., 1.]
    }).equals(missing_indicator_df)


def test_label_encode():
    le_df = novel.label_encode(
        df=pd.DataFrame({
            'a': [np.nan, 3., 2., 1.],
            'Indication': ['1', '0', '4', '0']
        }),
        multi_cat_levels={
            'a': (1., 2., 3.),
            'Indication': ('0', '4')
        },
        missing_indication_value='1'
    )
    assert pd.DataFrame({
        'a': [np.nan, 2., 1., 0.],
        'Indication': [np.nan, 0., 1., 0.]
    }).equals(le_df)


def test_preprocess_novel_pre_split(initial_df_permutations_fixture):
    """End-to-end test which just checks that dtypes and column names are as
        expected."""
    df = initial_df_permutations_fixture

    indications = get_indication_variable_names(df.columns)
    df = df[flatten_model_var_dict(NOVEL_MODEL_VARS) + indications]

    multi_category_levels: Dict[str, Tuple] = copy.deepcopy(
        MULTI_CATEGORY_LEVELS)
    multi_category_levels[INDICATION_VAR_NAME] = tuple(indications)

    df = preprocess_novel_pre_split(
        df,
        category_mapping={"S03ECG": {1.0: 0.0, 4.0: 1.0, 8.0: 1.0}},
        indication_variable_name=INDICATION_VAR_NAME,
        indications=indications,
        missing_indication_value=MISSING_IND_CATEGORY,
        multi_category_levels=multi_category_levels,
    )

    for column_name in df.columns:
        assert is_numeric_dtype(df[column_name])

    assert set(df.columns) == {
        "S03ASAScore",
        "S03CardiacSigns",
        "S03RespiratorySigns",
        "S03DiagnosedMalignancy",
        "S03Pred_Peritsoil",
        "S02PreOpCTPerformed",
        "S03ECG",
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
        "Target",
        "Indication"
    }


class TestWinsorizeNovel:
    @pytest.fixture()
    def input_df_fixture(self):
        return pd.DataFrame({
            'a': [0.0, 0.25, 0.5, 0.75, 1.0],
            'b': [1.0, 1.25, 1.5, 1.75, 2.0],
            'ignore': [0, 1, 0, 1, 0]
        })

    @pytest.fixture()
    def thresholds_fixture(self):
        return {'a': [0.2, 0.8], 'b': [None, 1.8]}

    @pytest.fixture()
    def output_df_fixture(self):
        return pd.DataFrame({
            'a': [0.2, 0.25, 0.5, 0.75, 0.8],
            'b': [1.0, 1.25, 1.5, 1.75, 1.8],
            'ignore': [0, 1, 0, 1, 0]
        })

    def test_winsorise_novel_quantiles_input(
        self, input_df_fixture, output_df_fixture, thresholds_fixture
    ):
        winsor_df, thresholds = novel.winsorize_novel(
            df=input_df_fixture,
            cont_vars=['a', 'b'],
            quantiles=(0.2, 0.8),
            include={'b': (False, True)}
        )
        assert thresholds == thresholds_fixture
        assert output_df_fixture.equals(winsor_df)

    def test_winsorise_novel_thresholds_input(
        self, input_df_fixture, output_df_fixture, thresholds_fixture
    ):
        winsor_df, thresholds = novel.winsorize_novel(
            df=input_df_fixture,
            thresholds=thresholds_fixture
        )
        assert thresholds == thresholds_fixture
        assert output_df_fixture.equals(winsor_df)
