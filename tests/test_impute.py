import numpy as np
import pandas as pd
from utils import impute


def test_determine_n_imputations(simple_df_with_missingness_fixture):
    n_imputations, fraction_incomplete = impute.determine_n_imputations(
        df=simple_df_with_missingness_fixture
    )
    assert fraction_incomplete == 0.4
    assert n_imputations == 40


class TestSplitterWinsorMICE:
    def test_placeholder(self):
        assert False


def test_find_missing_indices(simple_df_with_missingness_fixture):
    missing_i = impute.find_missing_indices(simple_df_with_missingness_fixture)
    assert {
        'a': np.array([2]),
        'b': np.array([1])
    } == missing_i

    complete_df = pd.DataFrame({'a': np.ones(4)})
    missing_i_if_complete = impute.find_missing_indices(complete_df)
    assert isinstance(missing_i_if_complete['a'], np.ndarray)
    assert missing_i_if_complete['a'].size == 0
