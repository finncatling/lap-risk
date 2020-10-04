import pytest
import numpy as np
import pandas as pd
from utils.simulate import simulate_initial_df
from utils import data_check


@pytest.fixture(scope='function')
def initial_df_fixture(initial_df_specification_fixture):
    return simulate_initial_df(
        specification=initial_df_specification_fixture,
        n_rows=100,
        n_hospitals=5,
        missing_frac=0.05,
        complete_indications=True,
        complete_target=True,
        complete_institution=True,
        round_1dp_variables=(
            "S03PreOpArterialBloodLactate",
            "S03Urea",
            "S03WhiteCellCount",
            "S03Potassium"
        ),
        random_seed=1
    )


def test_check_nela_data_column_names(
    initial_df_fixture: pd.DataFrame,
    initial_df_specification_fixture: dict
):
    """Test that check_nela_data_column_names() raises an AssertionError when
        passed a DataFrame with missing columns."""
    df_without_target_column = initial_df_fixture.drop(
        initial_df_specification_fixture['var_names']['target'],
        axis=1
    )
    with pytest.raises(AssertionError):
        data_check.check_nela_data_column_names(
            df_without_target_column,
            initial_df_specification_fixture
        )


def test_check_nela_data_types(initial_df_fixture: pd.DataFrame):
    """Test that check_nela_data_types() raises an AssertionError when
        passed a DataFrame containing a string."""
    initial_df_fixture.iloc[0, 0] = 'a string'
    with pytest.raises(AssertionError):
        data_check.check_nela_data_types(initial_df_fixture)


def test_check_nela_data_complete_columns(
    initial_df_fixture: pd.DataFrame,
    initial_df_specification_fixture: dict
):
    """Test that check_nela_data_complete_columns() raises an AssertionError
        when specific columns are incomplete."""
    initial_df_fixture.loc[
        0,
        initial_df_specification_fixture['var_names']['target']
    ] = np.nan
    with pytest.raises(AssertionError):
        data_check.check_nela_data_complete_columns(
            initial_df_fixture,
            initial_df_specification_fixture
        )


def test_check_nela_data_categories(
    initial_df_fixture: pd.DataFrame,
    initial_df_specification_fixture: dict
):
    """Test that check_nela_data_categories() raises an AssertionError
        when expected categories are absent from NELA data. initial_df_fixture
        only has a small number of rows so rare categories are absent."""
    with pytest.raises(AssertionError):
        data_check.check_nela_data_categories(
            initial_df_fixture,
            initial_df_specification_fixture
        )


def test_warn_if_high_nela_data_missingness(initial_df_fixture: pd.DataFrame):
    """Test that warning is raised whenever there high missingness in a NELA
        variable. Missingness is at least 50% for column 0 in this case."""
    half_n_rows = np.round(initial_df_fixture.shape[0] / 2).astype(int)
    initial_df_fixture.iloc[:half_n_rows, 0] = np.nan
    with pytest.warns(UserWarning):
        data_check.warn_if_high_nela_data_missingness(
            initial_df_fixture,
            warning_fraction=0.1
        )
