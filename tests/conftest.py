import numpy as np
import pandas as pd
import pytest

from utils.simulate import simulate_initial_df
from utils.data_check import get_initial_df_specification


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


# TODO: Consider widening scope from https://tinyurl.com/y6t7m77q
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
        n_rows=500,
        n_hospitals=70,
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
        random_seed=request.param
    )
