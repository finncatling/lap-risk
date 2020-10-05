from typing import Iterable, List

import pandas as pd

INDICATION_VAR_NAME = "Indication"
INDICATION_PREFIX = "S05Ind_"
MISSING_IND_CATEGORY = f"{INDICATION_PREFIX}Missing"


def ohe_to_single_column(
    df: pd.DataFrame, variable_name: str, categories: List[str]
) -> pd.DataFrame:
    """Changes a variable that is one-hot encoded over multiple DataFrame
        columns to integers in a single column."""
    df[variable_name] = df[categories].idxmax(axis=1)
    return df.drop(categories, axis=1)


def get_indication_variable_names(
    columns: Iterable[str],
    prefix: str = INDICATION_PREFIX
) -> List[str]:
    """Given an iterable of column names, isolates just those that are binary
        indication variables.

    Args:
        columns: Column names
        prefix: Prefix of the variables which are binary indications

    Returns:
        Binary indication variables
    """
    return [c for c in columns if prefix in c]


def get_common_single_indications(
    indication_df: pd.DataFrame,
    frequency_threshold: int
) -> List[str]:
    """Finds the 'common single indications' - those indications that occur in
        isolation >= frequency_threshold times

    Args:
        indication_df: Column names are individual indications, values are
            indicator variables so are integers in {0, 1}
        frequency_threshold:

    Returns:
        Common single indications in descending frequency order
    """
    single_indication_frequencies = indication_df.loc[
        indication_df.sum(1) == 1
    ].sum(0)
    return single_indication_frequencies.loc[
        single_indication_frequencies >= frequency_threshold
    ].sort_values(ascending=False).index.tolist()
