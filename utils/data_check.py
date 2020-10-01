import os
from typing import List
from warnings import warn

import numpy as np
import pandas as pd

from utils.constants import NELA_DATA_FILEPATH
from utils.inspect import percent_missing
from utils.io import load_object
from utils.model.novel import get_indication_variable_names


def get_initial_df_specification(
    specification_filepath: str = os.path.join(
        'config', 'initial_df_univariate_specification.pkl')
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

    # TODO: Check (with tolerance) prevalence of each category
    # TODO: Check (with tolerance) distribution of continuous variables

    Args:
        data_filepath: Path to .pkl containing NELA data
        missingness_warning_fraction: In interval [0, 1]. Shows the user for any
        column that has more than this fraction of missing values

    Returns:
        NELA data
    """
    df = pd.read_pickle(data_filepath).reset_index(drop=True)
    spec = get_initial_df_specification()

    # Make a flat list of names of all columns we expect in the NELA data
    all_spec_column_names: List[str] = (
        spec['var_names']['institutions'] +
        spec['var_names']['cat'] +
        spec['var_names']['cont'] +
        spec['var_names']['indications'] +
        [spec['var_names']['target']]  # this is a str. need to put it in a list
    )
    # Check that this column name list has the form we expect
    assert isinstance(all_spec_column_names, list)
    assert all(isinstance(col_name, str) for col_name in all_spec_column_names)
    # Check that the NELA data contains all and only those columns
    assert set(df.columns) == set(all_spec_column_names)

    # Check DataFrame contain only floats, ints and missing values
    assert np.isreal(df).all()

    # Check that there are no missing variables in some columns
    complete_columns = [
                           spec['var_names']['target'],
                           spec['var_names']['institutions'][0]
                       ] + get_indication_variable_names(df.columns)
    for column_name in complete_columns:
        assert percent_missing(df, column_name) == 0.

    # Check categories for each categorical variable
    for var_name, expected_probabilities in spec['cat_fits'].items():
        actual_probabilities = df[var_name].value_counts(
            normalize=True,
            dropna=True
        )
        # Check all and only expected categories are present
        assert (
            set(actual_probabilities.index) ==
            set(expected_probabilities.index)
        )

    # Issue a warning for any column which has high missingness
    missingness_warning_percent = 100 * missingness_warning_fraction
    for column_name in df.columns:
        column_percent_missing = percent_missing(df, column_name)
        if column_percent_missing > missingness_warning_percent:
            warn(f'{column_name} has '
                 f'{np.round(column_percent_missing, 1)}% missing values')

    return df
