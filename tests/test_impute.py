import numpy as np
import pandas as pd
import pytest

from utils import impute


def test_determine_n_imputations(simple_df_with_missingness_fixture):
    n_imputations, fraction_incomplete = impute.determine_n_imputations(
        df=simple_df_with_missingness_fixture
    )
    assert fraction_incomplete == 0.4
    assert n_imputations == 40


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


class TestImputationInfo:
    @pytest.fixture()
    def imputation_info_fixture(self, simple_df_with_missingness_fixture):
        ii = impute.ImputationInfo()
        ii.add_stage("1st description", simple_df_with_missingness_fixture)
        simple_df_with_missingness_fixture['high_missingness_column'] = [
            0., np.nan, np.nan, np.nan, 4.
        ]
        ii.add_stage("2nd description", simple_df_with_missingness_fixture)
        return ii

    def test_imputation_info(self, imputation_info_fixture):
        ii = imputation_info_fixture
        assert ii.descriptions == ["1st description", "2nd description"]
        assert ii.fraction_incomplete == [0.4, 0.6]
        assert ii.n_min_imputations == [40, 60]
        assert ii.n_imputations == [40, 80]
        assert ii.multiple_of_previous_n_imputations == [1, 2]
