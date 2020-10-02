import os
from typing import List
from warnings import warn

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils.constants import NELA_DATA_FILEPATH, ROOT_DIR
from utils.inspect import percent_missing
from utils.io import load_object
from utils.model.novel import get_indication_variable_names


def get_initial_df_specification(
    specification_filepath: str = os.path.join(
        ROOT_DIR, 'config', 'initial_df_univariate_specification.pkl')
) -> dict:
    """Specification for the continuous and categorical variables in the NELA
        data. Contains all the variables names, the categories (and
        associated probabilities) for each categorical variable, plus parameters
        for the parametric distribution that most closely fits the univariate
        empirical distributions of each continuous variable."""
    return load_object(specification_filepath)


def load_nela_data_and_sanity_check(
    data_filepath: str = NELA_DATA_FILEPATH,
    missingness_warning_fraction: float = 0.1
) -> pd.DataFrame:
    """Loads NELA data which has already undergone initial univariate
        wrangling and variable selection within the BDAU (in
        0_univariate_wrangling.ipynb). Checks that the data looks like we expect
        it to.

    Args:
        data_filepath: Path to .pkl containing NELA data
        missingness_warning_fraction: In interval [0, 1]. Shows warning for any
            column that has more than this fraction of missing values

    Returns:
        NELA data
    """
    df = pd.read_pickle(data_filepath).reset_index(drop=True)
    spec = get_initial_df_specification()

    check_nela_data_column_names(df, spec)
    check_nela_data_types(df)
    check_nela_data_complete_columns(df, spec)
    check_nela_data_categories(df, spec)
    warn_if_high_nela_data_missingness(df, missingness_warning_fraction)
    # TODO: Check (with tolerance) prevalence of each category
    # TODO: Check (with tolerance) distribution of continuous variables

    return df


def check_nela_data_column_names(df: pd.DataFrame, spec: dict) -> None:
    """Make a flat list of names of all columns we expect in the NELA data.
        Check that this column name list has the form we expect, then check
        that the NELA data contains all and only those columns.

    Args:
        df: NELA data
        spec: Initial data specification from get_initial_df_specification()
    """
    all_spec_column_names: List[str] = (
        spec['var_names']['institutions'] +
        spec['var_names']['cat'] +
        spec['var_names']['cont'] +
        spec['var_names']['indications'] +
        [spec['var_names']['target']]  # this is a str. need to put it in a list
    )
    assert isinstance(all_spec_column_names, list)
    assert all(isinstance(col_name, str) for col_name in all_spec_column_names)
    assert set(df.columns) == set(all_spec_column_names)


def check_nela_data_types(df: pd.DataFrame) -> None:
    """Check that NELA data contain only numeric dtypes. This check should pass
        with floats, ints and missing values.

    Args:
        df: NELA data
    """
    for column_name in df.columns:
        assert is_numeric_dtype(df[column_name])


def check_nela_data_complete_columns(df: pd.DataFrame, spec: dict) -> None:
    """Check that some specific columns in the NELA data (the target column,
        institution ID column and binary indication indicator columns)
        contain no missing values.

    Args:
        df: NELA data
        spec: Initial data specification from get_initial_df_specification()
    """
    complete_columns = [
        spec['var_names']['target'],
        spec['var_names']['institutions'][0]
    ] + get_indication_variable_names(df.columns)
    for column_name in complete_columns:
        assert percent_missing(df, column_name) == 0.


def check_nela_data_categories(df: pd.DataFrame, spec: dict) -> None:
    """Check that each categorical variable in the NELA contains all and only
        the categories we expect.

    Args:
        df: NELA data
        spec: Initial data specification from get_initial_df_specification()
    """
    for var_name, expected_probabilities in spec['cat_fits'].items():
        actual_probabilities = df[var_name].value_counts(
            normalize=True,
            dropna=True
        )
        assert (
            set(actual_probabilities.index) ==
            set(expected_probabilities.index)
        )


def warn_if_high_nela_data_missingness(
    df: pd.DataFrame,
    warning_fraction: float
) -> None:
    """Check that NELA data contain only numeric dtypes. This check should pass
        with floats, ints and missing values.

    Args:
        df: NELA data
        warning_fraction: In interval [0, 1]. Shows warning for any column that
            has more than this fraction of missing values
    """
    missingness_warning_percent = 100 * warning_fraction
    for column_name in df.columns:
        column_percent_missing = percent_missing(df, column_name)
        if column_percent_missing > missingness_warning_percent:
            warn(f'{column_name} has '
                 f'{np.round(column_percent_missing, 1)}% missing values')
    for column_name in df.columns:
        assert is_numeric_dtype(df[column_name])
