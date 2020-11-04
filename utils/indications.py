import re
from typing import Iterable, List, Dict, Tuple

import numpy as np
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


def ohe_single_indications(
    indication_df: pd.DataFrame,
    indication_subset_names: List[str]
) -> pd.DataFrame:
    """Makes a new one-hot-encoded DataFrame whose column names are
        indication_subset_names. Values are 1 in cases where that indication
        occurs in isolation, otherwise 0"""
    ohe_indication_df = pd.DataFrame(
        data=np.zeros(
            (indication_df.shape[0], len(indication_subset_names))
        ).astype('int64'),
        columns=indication_subset_names
    )
    for name in indication_subset_names:
        ohe_indication_df.loc[
            ((indication_df.sum(1) == 1) & (indication_df[name] == 1)),
            name
        ] = 1

    # Check that no cases have more than one category encoded
    assert not ohe_indication_df.loc[ohe_indication_df.sum(1) > 1].shape[0]

    return ohe_indication_df


def report_ohe_category_assignment(
    data: pd.DataFrame,
    category_name: str
) -> None:
    """Given a DataFrame where the columns one-hot-encode a categorical
        variable, reports the number / proportion of rows where a category is
        assigned."""
    n_cases_total = data.shape[0]
    n_cases_assigned = data.loc[data.sum(1) == 1].shape[0]
    print(
        f"{n_cases_assigned} cases out of {n_cases_total} ("
        f"{100 * np.round(n_cases_assigned / n_cases_total, 3)}%) now have "
        f"an assigned {category_name}"
    )


class IndicationNameProcessor:
    """Process indication names for use in model specification and plotting."""
    def __init__(
        self,
        multi_category_levels: Dict[str, Tuple],
        max_line_length: int = 12,
        indication_var_name: str = INDICATION_VAR_NAME,
        indication_prefix: str = INDICATION_PREFIX,
        missing_indication_category: str = MISSING_IND_CATEGORY,
        remove_missing_category: bool = True,
    ):
        self.multi_category_levels = multi_category_levels
        self.max_line_length = max_line_length
        self.ind_var_name = indication_var_name
        self.ind_prefix = indication_prefix
        self.missing_ind_category = missing_indication_category
        self.remove_missing_category = remove_missing_category

    @property
    def names(self) -> List[str]:
        names = list(self.multi_category_levels[self.ind_var_name])
        if self.remove_missing_category:
            names.remove(self.missing_ind_category)
        return names

    @property
    def sanitized(self) -> List[str]:
        return [self._sanitize_indication(ind_name) for ind_name in self.names]

    def _sanitize_indication(self, ind: str) -> str:
        """Make individual indication names more human-readable when plotted."""
        ind = ind[len(self.ind_prefix):]
        ind_word_list = re.findall("[A-Z][^A-Z]*", ind)
        sanitized_list = [ind_word_list[0]]
        for word in ind_word_list[1:]:
            last_2_words_joined = f'{sanitized_list[-1]} {word}'
            if len(last_2_words_joined) > self.max_line_length:
                sanitized_list.append(word)
            else:
                sanitized_list[-1] = last_2_words_joined
        sanitized = "\n".join(sanitized_list)
        sanitized = sanitized.lower()
        return sanitized[0].upper() + sanitized[1:]
