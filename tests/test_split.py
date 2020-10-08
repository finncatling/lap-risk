import numpy as np
import pandas as pd
import pytest

from utils import split


def test_drop_incomplete_cases(simple_df_with_missingness_fixture):
    complete_df, drop_stats = split.drop_incomplete_cases(
        simple_df_with_missingness_fixture
    )
    assert drop_stats['n_total_cases'] == 5
    assert drop_stats['n_complete_cases'] == 3
    assert drop_stats['n_dropped_cases'] == 2
    assert drop_stats['fraction_dropped'] == 0.4
    assert pd.DataFrame({
        'a': [0., 3., 4.],
        'b': [0., 3., 4.]
    }, index=[0, 3, 4]).equals(complete_df)
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
        assert pd.DataFrame({
            'a': [0., 0., 1., np.nan, 1.],
            'b': [1.6, 3.8, np.nan, np.nan, 9.1],
            'institution': [0, 0, 1, 2, 3]
        }, index=[0, 1, 2, 3, 4]).equals(post_split_fixture.df)

    def test_test_institution_ids(self, post_split_fixture):
        """Due to randomisation we only know the contents of the inner arrays
            a posteriori, but once these are known we should be able to work
            out the results of the later tests a priori"""
        assert (post_split_fixture.test_institution_ids ==
                [np.array([3]), np.array([0])])

    def test_train_institution_ids(self, post_split_fixture):
        assert isinstance(post_split_fixture.train_institution_ids, list)
        # outer list cast to numpy array for convenient use of .all()
        assert (np.array(post_split_fixture.train_institution_ids) ==
                np.array([[0, 1, 2], [1, 2, 3]])).all()

    def test_test_i(self, post_split_fixture):
        """Piecewise comparison as structure (list of numpy arrays of different
            lengths) complicates use of all()"""
        assert isinstance(post_split_fixture.test_i, list)
        assert len(post_split_fixture.test_i) == 2
        assert post_split_fixture.test_i[0] == np.array([4])
        assert (post_split_fixture.test_i[1] == np.array([0, 1])).all()

    def test_train_i(self, post_split_fixture):
        assert isinstance(post_split_fixture.train_i, list)
        assert len(post_split_fixture.train_i) == 2
        assert (post_split_fixture.train_i[0] == np.array([0, 1, 2, 3])).all()
        assert (post_split_fixture.train_i[1] == np.array([2, 3, 4])).all()

    def test_split_stats(self, post_split_fixture):
        assert post_split_fixture.split_stats == {
            "n_test_cases": [1, 2],
            "test_fraction_of_complete_cases": [1 / 3, 2 / 3],
            "test_fraction_of_total_cases": [0.2, 0.4],
        }


@pytest.fixture(scope='function')
def splitter_current_model_df_fixture() -> pd.DataFrame:
    """Note discontinuous index, as if incomplete cases have been previously
        dropped."""
    return pd.DataFrame({
        'target': [0, 0, 1, 0],
        'a': [0., 0., 4., 2.],
    }, index=[1, 3, 4, 5])


def test_split_into_folds(splitter_current_model_df_fixture):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        n_total_train_cases,
        n_intersection_train_cases
    ) = split.split_into_folds(
        df=splitter_current_model_df_fixture,
        indices={
            'train': np.array([0, 2, 3, 4]),
            'test': np.array([1, 5])
        },
        target_var_name='target'
    )
    assert all(pd.DataFrame({'a': [0., 4.]}) == X_train)
    assert (np.array([0, 1]) == y_train).all()
    assert all(pd.DataFrame({'a': [0., 2.]}) == X_test)
    assert (np.array([0, 0]) == y_test).all()
    assert n_total_train_cases == 4
    assert n_intersection_train_cases == 2


def test_split_into_folds_index_assertion(splitter_current_model_df_fixture):
    with pytest.raises(AssertionError):
        _ = split.split_into_folds(
            df=splitter_current_model_df_fixture.reset_index(drop=True),
            indices={
                'train': np.array([0, 2, 3, 4]),
                'test': np.array([1, 5])
            },
            target_var_name='target'
        )
