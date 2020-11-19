from utils import wrangling


def test_percent_missing(simple_df_with_missingness_fixture):
    mp = wrangling.percent_missing(
        df=simple_df_with_missingness_fixture,
        col_name='a'
    )
    assert mp == 20.
