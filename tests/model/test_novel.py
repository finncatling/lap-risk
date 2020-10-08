import pytest
import numpy as np
import pandas as pd

from utils.model import novel


@pytest.fixture()
def missing_categories_df_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        'a': [1., 2., 3., np.nan],
        'b': [4., 5., 4., 6.]
    })


def test_combine_categories(missing_categories_df_fixture):
    combined_df = novel.combine_categories(
        df=missing_categories_df_fixture,
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
    assert pd.DataFrame({
        'a': [1., 2., 2., np.nan],
        'b': [6., 5., 6., 6.]
    }).equals(combined_df)


def test_add_missingness_indicators(missing_categories_df_fixture):
    missing_indicator_df = novel.add_missingness_indicators(
        df=missing_categories_df_fixture,
        variables=['a']
    )
    assert pd.DataFrame({
        'a': [1., 2., 3., np.nan],
        'b': [4., 5., 4., 6.],
        'a_missing': [0., 0., 0., 1.]
    }).equals(missing_indicator_df)


def test_preprocess_novel_pre_split():
    assert False


def test_label_encode():
    assert False
