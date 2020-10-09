from utils import impute


class TestImputationInfo:
    def test_add_stage(self):
        assert False

    def test__sanity_check(self):
        assert False

    def test__determine_adjusted_n_imputations(self):
        assert False


def test_determine_n_imputations(simple_df_with_missingness_fixture):
    n_imputations, fraction_incomplete = impute.determine_n_imputations(
        df=simple_df_with_missingness_fixture
    )
    assert fraction_incomplete == 0.4
    assert n_imputations == 40
