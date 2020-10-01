import pytest
import numpy as np
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


class TestTrainTestSplitter:
    @pytest.fixture(scope='class')
    def df_for_train_test_split_fixture(self) -> pd.DataFrame:
        return pd.DataFrame({
            'institution': [0, 0, 1, 2, 3],
            'a': [0., 0., 1., np.nan, 1.],
            'b': [1.6, 3.8, np.nan, np.nan, 9.1],
            'c': [np.nan, 1., 2., np.nan, 2.]
        }, index=[0, 1, 3, 4, 5])

    @pytest.fixture(scope='class')
    def post_split_fixture(self, df_for_train_test_split_fixture):
        tts = split.TrainTestSplitter(
            df=df_for_train_test_split_fixture,
            split_variable_name='institution',
            test_fraction=0.25,
            n_splits=2,
            current_nela_model_vars=['a', 'b'],
            random_seed=1
        )
        tts.split()
        return tts

    def test_n_institutions(self, post_split_fixture):
        assert post_split_fixture.n_institutions == 4

    def test_n_test_institutions(self, post_split_fixture):
        assert post_split_fixture.n_test_institutions == 1

    def test_n_train_institutions(self, post_split_fixture):
        assert post_split_fixture.n_train_institutions == 3

    def test_institution_ids(self, post_split_fixture):
        assert all(post_split_fixture.institution_ids == np.array(
            [0, 1, 2, 3]))

    def test__preprocess_df(self, post_split_fixture):
        assert all(post_split_fixture.df == pd.DataFrame({
            'a': [0., 0., 1., np.nan, 1.],
            'b': [1.6, 3.8, np.nan, np.nan, 9.1],
            'institution': [0, 0, 1, 2, 3]
        }, index=[0, 1, 2, 3, 4]))

    def test_test_institution_ids(self, post_split_fixture):
        assert False

    def test_split(self):
        assert False

    # def test__split_institutions(self):
    #     assert False
    #
    # def test__split_cases(self):
    #     assert False
    #
    # def test__calculate_split_stats(self):
    #     assert False


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

