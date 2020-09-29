import os
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from utils.io import load_object
from utils.split import TrainTestSplitter
from utils.model.novel import NOVEL_MODEL_VARS, get_indication_variable_names
from utils.simulate import simulate_initial_df


@pytest.fixture(scope='session')
def initial_df_specification(
    specification_filepath: str = os.path.join(
        os.pardir, 'config', 'initial_df_univariate_specification.pkl')
) -> dict:
    """Specification for the continuous and categorical variables in the NELA
        data. Contains all the variables names, the categories (and
        associated probabilities) for each categorical variable, plus parameters
        for the parametric distribution that most closely fits the univariate
        empirical distributions of each continuous variable."""
    return load_object(specification_filepath)


# TODO: Consider widening scope from https://tinyurl.com/y6t7m77q
@pytest.fixture(scope="function", params=[1, 2, 3])
def initial_df_fixture(
    request,
    initial_df_specification: dict,
    n_rows: int = 500,
    n_hospitals: int = 70,
    missing_frac: float = 0.05,
    complete_indications: bool = True,
    complete_target: bool = True,
    complete_institution: bool = True,
) -> pd.DataFrame:
    """Simulates NELA data after initial univariate wrangling and variable
        selection (i.e. the output of 0_univariate_wrangling.ipynb), which is
        input to 01_train_test_split.py

    Args:
        request: Used by pytest to pass in different random seeds, generating
            different versions of the fixture
        initial_df_specification: Specification for the continuous and
            categorical variables in the NELA data
        n_rows: Number of rows in DataFrame
        n_hospitals: Number of hospitals in DataFrame
        missing_frac: Fraction of data that is missing (for incomplete
            variables)
        complete_indications: if True, don't introduce missingness into the
            binary indications variables
        complete_target: if True, don't introduce missingness into the
            target variable
        complete_institution: if True, don't introduce missingness into the
            hospital / trust ID variable

    Returns:
        Simulated NELA data
    """
    return simulate_initial_df(
        specification=initial_df_specification,
        n_rows=n_rows,
        n_hospitals=n_hospitals,
        missing_frac=missing_frac,
        complete_indications=complete_indications,
        complete_target=complete_target,
        complete_institution=complete_institution,
        random_seed=request.param
    )


# TODO: Consider correct scope from https://tinyurl.com/y6t7m77q
@pytest.fixture(scope="function")
def train_test_split_fixture(initial_df_fixture):
    model_vars = NOVEL_MODEL_VARS["cat"] + NOVEL_MODEL_VARS["cont"]
    tts = TrainTestSplitter(
        df=initial_df_fixture,
        split_variable_name="HospitalId.anon",
        test_fraction=0.2,
        n_splits=5,
        current_nela_model_vars=list(model_vars),
        random_seed=5
    )
    tts.split()
    return tts
