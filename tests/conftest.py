import os
import numpy as np
import pandas as pd
import pytest
from typing import Tuple

from utils.io import load_object
from utils.split import TrainTestSplitter
from utils.model.novel import NOVEL_MODEL_VARS, get_indication_variable_names


# TODO: Separate out initial DataFrame spec as separate fixture


# TODO: Consider correct scope from https://tinyurl.com/y6t7m77q
@pytest.fixture(scope="function", params=[1, 2, 3])
def initial_df_fixture(
    request,
    n_rows: int = 1000,
    n_hospitals: int = 170,
    missing_frac: float = 0.05,
    complete_indications: bool = True,
    complete_target: bool = True,
    complete_institution: bool = True,
    specification_filepath: str = os.path.join(
        os.pardir, 'config', 'initial_df_univariate_specification.pkl')
) -> pd.DataFrame:
    """Simulates NELA data after initial univariate wrangling and variable
        selection (i.e. the output of 0_univariate_wrangling.ipynb), which is
        input to 01_train_test_split.py

    Args:
        request: Used by pytest to pass in different random seeds, generating
            different versions of the fixture
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
        specification_filepath:

    Returns:
        Simulated NELA data
    """
    rnd = np.random.RandomState(request.param)
    spec = load_object(specification_filepath)
    df = pd.DataFrame()

    # Create institution (hospital or trust) ID column
    df[spec['var_names']['institutions'][0]] = rnd.randint(
        n_hospitals, size=n_rows)

    # Create other categorical columns
    for var_name, probabilities in spec['cat_fits'].items():
        cat_samples_i_2d = np.random.multinomial(
            n=1,
            pvals=probabilities.values,
            size=n_rows)
        cat_samples_i_1d = np.argmax(cat_samples_i_2d, 1)
        cat_samples = [probabilities.index[i] for i in cat_samples_i_1d]
        df[var_name] = cat_samples

    # Create continuous columns
    # TODO: Update this
    for i in NOVEL_MODEL_VARS["cont"]:
        center = rnd.randint(50, 150)
        df[i] = rnd.normal(center, 10, n_rows)

    for i in NOVEL_MODEL_VARS["cat"]:
        cats = (1.0, 2.0, 4.0, 8.0)
        df[i] = rnd.choice(cats, n_rows)

    # Make list of columns which will have missing values
    missing_columns = df.columns.tolist()
    if complete_indications:
        for c in get_indication_variable_names(df.columns):
            missing_columns.remove(c)
    if complete_target:
        missing_columns.remove(spec['var_names']['target'])
    if complete_institution:
        missing_columns.remove(spec['var_names']['institutions'][0])

    # Introduce missing values
    for col in missing_columns:
        df.loc[df.sample(
            frac=missing_frac,
            random_state=rnd
        ).index, col] = np.nan

    return df


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
