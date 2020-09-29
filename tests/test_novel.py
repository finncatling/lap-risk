from utils.model import novel


def test_combine(initial_df_fixture):
    mapping = {"S03ECG": {
        1: 1,
        2: 2,
        4: 4,
        8: 4,
    }}
    combined_df = novel.combine_categories(initial_df_fixture, mapping)
    assert initial_df_fixture["S03ECG"].unique != combined_df["S03ECG"].unique


def test_add_missingness_indicators(initial_df_fixture):
    cols = initial_df_fixture.shape[1]
    df2 = novel.add_missingness_indicators(
        initial_df_fixture,
        ["S01AgeOnArrival", "S03SerumCreatinine"]
    )
    cols = df2.shape[1] - cols
    assert cols == 2
