import numpy as np
import pandas as pd
import pytest

from utils import split
from utils.model.novel import NOVEL_MODEL_VARS


# TODO: Consider correct scope from https://tinyurl.com/y6t7m77q
@pytest.fixture(scope="function")
def initial_df_fixture(n_rows: int = 1000):
    """Simulates NELA data after initial univariate wrangling and variable
        selection (i.e. the output of 0_univariate_wrangling.ipynb), which is
        input to 01_train_test_split.py"""
    df = pd.DataFrame()

    # create continuous columns
    for i in NOVEL_MODEL_VARS["cont"]:
        center = np.random.randint(50, 150)
        df[i] = np.random.normal(center, 10, n_rows)

    for i in NOVEL_MODEL_VARS["cat"]:
        cats = (1.0, 2.0, 4.0, 8.0)
        df[i] = np.random.choice(cats, n_rows)

    # introduce missingness
    for col in df.columns:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    df["HospitalId.anon"] = np.random.randint(170, size=n_rows)
    df["Target"] = np.random.randint(2, size=n_rows)
    return df


# TODO: Consider correct scope from https://tinyurl.com/y6t7m77q
@pytest.fixture(scope="function")
def train_test_split_fixture(initial_df_fixture):
    model_vars = NOVEL_MODEL_VARS["cat"] + NOVEL_MODEL_VARS["cont"]
    tts = split.TrainTestSplitter(
        df=initial_df_fixture,
        split_variable_name="HospitalId.anon",
        test_fraction=0.2,
        n_splits=5,
        current_nela_model_vars=list(model_vars),
        random_seed=5
    )
    tts.split()
    return tts
