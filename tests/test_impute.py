from utils import impute


def test_determine_n_imputations(simple_df_with_missingness_fixture):
    n_imputations, fraction_incomplete = impute.determine_n_imputations(
        df=simple_df_with_missingness_fixture
    )
    assert fraction_incomplete == 0.4
    assert n_imputations == 40


class TestSplitterWinsorMICE:
    def test_placeholder():
        assert False


def test_find_missing_indices():
    assert False
