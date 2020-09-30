import pandas as pd

from utils import split
from utils.model.novel import NOVEL_MODEL_VARS


def test_drop_incomplete_cases(simple_df_with_missingness_fixture):
    complete_df, drop_stats = split.drop_incomplete_cases(
        simple_df_with_missingness_fixture
    )
    assert drop_stats['n_total_cases'] == 5
    assert drop_stats['n_complete_cases'] == 3
    assert drop_stats['n_dropped_cases'] == 2
    assert drop_stats['fraction_dropped'] == 0.4
    assert all(complete_df == pd.DataFrame({
        'a': [0., 3., 4.],
        'b': [0., 3., 4.]
    }, index=[0, 3, 4]))
    # check input DataFrame not changed
    assert simple_df_with_missingness_fixture.shape == (5, 2)


def test_split_into_folds(initial_df_fixture):
    # TODO this test should probably be more comprehensive but it's passing
    #  for now
    indices = {
        'train': initial_df_fixture.sample(frac=0.6).index,
        'test': initial_df_fixture.sample(frac=0.2).index
    }
    stuff = split.split_into_folds(
        initial_df_fixture,
        indices,
        NOVEL_MODEL_VARS["target"]
    )
    assert stuff[0].shape[0] == initial_df_fixture.sample(frac=0.6).shape[0]


def test_train_test_split(train_test_split_fixture):
    assert len(train_test_split_fixture.train_institution_ids) == 5
