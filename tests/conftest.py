import numpy as np
import pandas as pd
import pytest

from utils.simulate import simulate_initial_df
from utils.data_check import get_initial_df_specification
from utils.split import TrainTestSplitter


@pytest.fixture(scope='function')
def simple_df_with_missingness_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        'a': [0., 1., np.nan, 3., 4.],
        'b': [0., np.nan, 2., 3., 4.]
    })


@pytest.fixture(scope='session')
def initial_df_specification_fixture() -> dict:
    """Specification for the continuous and categorical variables in the NELA
        data. Contains all the variables names, the categories (and
        associated probabilities) for each categorical variable, plus parameters
        for the parametric distribution that most closely fits the univariate
        empirical distributions of each continuous variable."""
    return get_initial_df_specification()


@pytest.fixture(scope="function", params=[1, 2, 3])
def initial_df_permutations_fixture(
    request,
    initial_df_specification_fixture,
) -> pd.DataFrame:
    """Simulates NELA data after initial univariate wrangling and variable
        selection (i.e. the output of 0_univariate_wrangling.ipynb), which is
        input to 01_train_test_split.py

    Args:
        request: Used by pytest to pass in different random seeds, generating
            different versions of the fixture
        initial_df_specification_fixture: Specification for the initial NELA
            data

    Returns:
        Simulated NELA data
    """
    return simulate_initial_df(
        specification=initial_df_specification_fixture,
        n_rows=600,
        n_hospitals=70,
        missing_frac=0.02,
        complete_indications=True,
        complete_target=True,
        complete_institution=True,
        round_1dp_variables=(
            "S03PreOpArterialBloodLactate",
            "S03Urea",
            "S03WhiteCellCount",
            "S03Potassium"
        ),
        random_seed=request.param
    )


@pytest.fixture(scope='class')
def df_for_train_test_split_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        'institution': [0, 0, 1, 2, 3],
        'a': [0., 0., 1., np.nan, 1.],
        'b': [1.6, 3.8, np.nan, np.nan, 9.1],
        'c': [np.nan, 1., 2., np.nan, 2.]
    }, index=[0, 1, 3, 4, 5])


@pytest.fixture(scope='class')
def train_test_split_fixture(df_for_train_test_split_fixture):
    tts = TrainTestSplitter(
        df=df_for_train_test_split_fixture,
        split_variable_name='institution',
        test_fraction=0.25,
        n_splits=2,
        current_nela_model_vars=['a', 'b'],
        random_seed=1
    )
    tts.split()
    return tts
