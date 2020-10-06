from typing import Tuple, List, Set

import numpy as np
import pandas as pd
import pytest

from utils import indications


@pytest.fixture()
def columns_and_indication_subset() -> Tuple[List[str], Set[str]]:
    return (
        [
            'S03PreOpArterialBloodLactate',
            'S03PreOpLowestAlbumin',
            'S03Sodium',
            'S03Pulse',
            'S03SystolicBloodPressure',
            'S05Ind_Peritonitis',
            'S05Ind_Perforation',
            'S05Ind_AbdominalAbscess',
        ],
        {
            'S05Ind_Peritonitis',
            'S05Ind_Perforation',
            'S05Ind_AbdominalAbscess',
        }
    )


def test_get_indication_variable_names_with_list(columns_and_indication_subset):
    columns, indication_columns = columns_and_indication_subset
    output = indications.get_indication_variable_names(columns)
    assert isinstance(output, list)
    assert set(output) == indication_columns


def test_get_indication_variable_names_with_pandas_columns(
    columns_and_indication_subset
):
    columns, indication_columns = columns_and_indication_subset
    df = pd.DataFrame(data=np.zeros((2, len(columns))),
                      columns=columns)
    output = indications.get_indication_variable_names(df.columns)
    assert isinstance(output, list)
    assert set(output) == indication_columns


@pytest.fixture()
def indications_df_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        'S05Ind_0': [1, 1, 1, 1, 0, 0, 0, 0],
        'S05Ind_1': [0, 0, 1, 0, 0, 0, 0, 0],
        'S05Ind_2': [0, 0, 1, 0, 0, 0, 0, 0],
        'S05Ind_3': [0, 0, 0, 1, 0, 1, 1, 1]
    })


@pytest.fixture()
def common_single_indications_fixture() -> List[str]:
    return ['S05Ind_3', 'S05Ind_0']


@pytest.fixture()
def ohe_single_indications_df_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        'S05Ind_3': [0, 0, 0, 0, 0, 1, 1, 1],
        'S05Ind_0': [1, 1, 0, 0, 0, 0, 0, 0]
    })


def test_get_common_single_indications(
    indications_df_fixture,
    common_single_indications_fixture
):
    common_single_inds = indications.get_common_single_indications(
        indication_df=indications_df_fixture,
        frequency_threshold=2
    )
    assert common_single_inds == common_single_indications_fixture


def test_ohe_single_indications(
    indications_df_fixture,
    common_single_indications_fixture,
    ohe_single_indications_df_fixture
):
    ohe_ind_df = indications.ohe_single_indications(
        indication_df=indications_df_fixture,
        indication_subset_names=common_single_indications_fixture
    )
    assert all(ohe_single_indications_df_fixture == ohe_ind_df)
