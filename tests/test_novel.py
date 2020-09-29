from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import pytest

from utils.model import novel


def test_combine(initial_df_fixture):
    mapping = {"S03ECG": {
        1: 1,
        2: 2,
        4: 4,
        8: 4,
    }}
    combined_df = novel.combine_categories(initial_df_fixture, mapping)
    assert initial_df_fixture["S03ECG"].unique != combined_df["S03ECG"].unique


def test_add_missingness_indicators(initial_df_fixture):
    cols = initial_df_fixture.shape[1]
    df2 = novel.add_missingness_indicators(
        initial_df_fixture,
        ["S01AgeOnArrival", "S03SerumCreatinine"]
    )
    cols = df2.shape[1] - cols
    assert cols == 2


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
    output = novel.get_indication_variable_names(columns)
    assert isinstance(output, list)
    assert set(output) == indication_columns


def test_get_indication_variable_names_with_pandas_columns(
    columns_and_indication_subset
):
    columns, indication_columns = columns_and_indication_subset
    df = pd.DataFrame(data=np.zeros((2, len(columns))),
                      columns=columns)
    output = novel.get_indication_variable_names(df.columns)
    assert isinstance(output, list)
    assert set(output) == indication_columns
