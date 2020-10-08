import numpy as np
import pandas as pd

from utils.model import novel


def test_combine_categories():
    combined_df = novel.combine_categories(
        df=pd.DataFrame({
            'a': [1., 2., 3., np.nan],
            'b': [4., 5., 4., 6.]
        }),
        category_mapping={
            'a': {
                1.: 1.,
                2.: 2.,
                3.: 2.
            },
            'b': {
                4.: 6.,
                5.: 5.,
                6.: 6.
            },
        }
    )
    assert all(pd.DataFrame({
        'a': [1., 2., 2., np.nan],
        'b': [6., 5., 6., 6.]
    }) == combined_df)


def test_add_missingness_indicators(initial_df_permutations_fixture):
    cols = initial_df_permutations_fixture.shape[1]
    df2 = novel.add_missingness_indicators(
        initial_df_permutations_fixture,
        ["S01AgeOnArrival", "S03SerumCreatinine"]
    )
    cols = df2.shape[1] - cols
    assert cols == 2


def test_preprocess_novel_pre_split():
    assert False
