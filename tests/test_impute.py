from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from utils import impute
from utils.split import TrainTestSplitter


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


@pytest.fixture(scope='module')
def df_fixture() -> pd.DataFrame:
    n_rows = 20
    rnd = np.random.RandomState(1)
    df = pd.DataFrame({'cont': np.linspace(-1, 1, num=n_rows)})
    df['bin'] = rnd.choice(2, size=n_rows)
    df['multicat'] = rnd.choice(3, size=n_rows)
    df['target'] = rnd.binomial(
        n=1, p=expit(df.cont + df.bin + (df.multicat - 1))
    )
    df.loc[1, 'cont'] = np.nan
    df.loc[8, 'bin'] = np.nan
    df.loc[(3, 10), 'multicat'] = np.nan
    return df


@pytest.fixture(scope='module')
def mock_train_test_split_fixture(df_fixture) -> TrainTestSplitter:
    """We use a mock to allow deterministic splitting"""
    even_i = np.arange(0, df_fixture.shape[0] - 1, 2)
    odd_i = np.arange(1, df_fixture.shape[0], 2)
    tts = mock.create_autospec(TrainTestSplitter)
    tts.train_i = [even_i, odd_i]
    tts.test_i = [odd_i, even_i]
    tts.n_splits = len(tts.train_i)
    return tts


@pytest.fixture(scope='module')
def splitter_winsor_mice_fixture(
    df_fixture, mock_train_test_split_fixture
) -> impute.SplitterWinsorMICE:
    swm = impute.SplitterWinsorMICE(
        df=df_fixture.drop('multicat', axis=1),
        train_test_splitter=mock_train_test_split_fixture,
        target_variable_name='target',
        cont_variables=['cont'],
        binary_variables=['bin'],
        winsor_quantiles=(0.01, 0.99),
        winsor_include=None,
        n_mice_imputations=2,
        n_mice_burn_in=1,
        n_mice_skip=1,
        random_seed=1
    )
    swm.split_winsorize_mice()
    return swm


@pytest.fixture(scope='module')
def categorical_imputer_fixture(
    df_fixture, splitter_winsor_mice_fixture
) -> impute.CategoricalImputer:
    cat_imputer = impute.CategoricalImputer(
        df=df_fixture,
        splitter_winsor_mice=splitter_winsor_mice_fixture,
        cat_vars=['multicat'],
        random_seed=1
    )
    cat_imputer.impute()
    return cat_imputer


class TestSplitterWinsorMICE:
    def test_all_vars(self, splitter_winsor_mice_fixture):
        assert splitter_winsor_mice_fixture.all_vars == [
            'cont', 'bin', 'target'
        ]

    def test_winsor_thresholds(self, splitter_winsor_mice_fixture):
        for i in range(splitter_winsor_mice_fixture.n_mice_imputations):
            assert all(
                [-1 < threshold < 1 for threshold in
                 splitter_winsor_mice_fixture.winsor_thresholds[i]['cont']]
            )

    def test_missing_i(self, splitter_winsor_mice_fixture):
        """Tested using loop as direct equality test runs into 'the truth
            value of an empty ndarray is ambiguous' errors."""
        assert (
            set(splitter_winsor_mice_fixture.missing_i.keys()) ==
            {'train', 'test'}
        )
        for fold_name, split in splitter_winsor_mice_fixture.missing_i.items():
            assert set(split.keys()) == {0, 1}
            for split_i, variables in split.items():
                assert set(variables.keys()) == {'cont', 'bin', 'target'}
                for variable_name, missing_i in variables.items():
                    if ((
                        fold_name == 'train' and
                        split_i == 0 and
                        variable_name == 'bin'
                    ) or (
                        fold_name == 'test' and
                        split_i == 1 and
                        variable_name == 'bin'
                    )):
                        assert missing_i == np.array([4], dtype='int64')
                    elif ((
                              fold_name == 'train' and
                              split_i == 1 and
                              variable_name == 'cont'
                          ) or (
                              fold_name == 'test' and
                              split_i == 0 and
                              variable_name == 'cont'
                          )):
                        assert missing_i == np.array([0], dtype='int64')
                    else:
                        assert missing_i.size == 0

    def test_imputed(self, splitter_winsor_mice_fixture):
        """Tested using loop as direct equality test runs into 'the truth
            value of an empty ndarray is ambiguous' errors."""
        assert (
            set(splitter_winsor_mice_fixture.imputed.keys()) ==
            {'train', 'test'}
        )
        for fold_name, split in splitter_winsor_mice_fixture.imputed.items():
            assert set(split.keys()) == {0, 1}
            for split_i, imputations in split.items():
                assert set(imputations.keys()) == {0, 1}
                for imp_i, variables in imputations.items():
                    assert set(variables.keys()) == {'cont', 'bin', 'target'}
                    for variable_name, imputed in variables.items():
                        if ((
                            fold_name == 'train' and
                            split_i == 0 and
                            variable_name == 'bin'
                        ) or (
                            fold_name == 'test' and
                            split_i == 1 and
                            variable_name == 'bin'
                        )):
                            assert imputed.size == 1
                            assert imputed[0] in {0, 1}
                        elif ((
                                  fold_name == 'train' and
                                  split_i == 1 and
                                  variable_name == 'cont'
                              ) or (
                                  fold_name == 'test' and
                                  split_i == 0 and
                                  variable_name == 'cont'
                              )):
                            assert imputed.size == 1
                            assert -1 < imputed[0] < 1
                        else:
                            assert imputed.size == 0

    def test_get_imputed_variables(self, splitter_winsor_mice_fixture):
        train_imputed = splitter_winsor_mice_fixture.get_imputed_variables(
            fold_name='train',
            split_i=0,
            imp_i=0
        )
        assert set(train_imputed.columns) == {'cont', 'bin', 'target'}
        train_unimputed, _ = splitter_winsor_mice_fixture._split_then_join_Xy(0)
        assert train_imputed[['cont', 'target']].equals(
            train_unimputed[['cont', 'target']]
        )
        assert train_unimputed.loc[train_unimputed.bin.notnull(), 'bin'].equals(
            train_imputed.loc[train_unimputed.bin.notnull(), 'bin']
        )
        imputed = train_imputed.loc[train_unimputed.bin.isnull(), 'bin'].values
        assert imputed.size == 1
        assert imputed[0] in {0, 1}



class TestCategoricalImputer:
    def test_missing_i(self, categorical_imputer_fixture):
        """Tested using loop as direct equality test runs into 'the truth
            value of an empty ndarray is ambiguous' errors."""
        assert (
            set(categorical_imputer_fixture.missing_i.keys()) ==
            {'train', 'test'}
        )
        for fold_name, split in categorical_imputer_fixture.missing_i.items():
            assert set(split.keys()) == {0, 1}
            for split_i, variables in split.items():
                assert set(variables.keys()) == {'multicat'}
                for variable_name, missing_i in variables.items():
                    if ((
                        fold_name == 'train' and
                        split_i == 0 and
                        variable_name == 'multicat'
                    ) or (
                        fold_name == 'test' and
                        split_i == 1 and
                        variable_name == 'multicat'
                    )):
                        assert missing_i == np.array([5], dtype='int64')
                    elif ((
                              fold_name == 'train' and
                              split_i == 1 and
                              variable_name == 'multicat'
                          ) or (
                              fold_name == 'test' and
                              split_i == 0 and
                              variable_name == 'multicat'
                          )):
                        assert missing_i == np.array([1], dtype='int64')
                    else:
                        assert missing_i.size == 0

    def test_imputed(self, categorical_imputer_fixture):
        """Tested using loop as direct equality test runs into 'the truth
            value of an empty ndarray is ambiguous' errors."""
        assert (
            set(categorical_imputer_fixture.imputed.keys()) ==
            {'train', 'test'}
        )
        for fold_name, split in categorical_imputer_fixture.imputed.items():
            assert set(split.keys()) == {0, 1}
            for split_i, imputations in split.items():
                assert set(imputations.keys()) == {0, 1}
                for imp_i, variables in imputations.items():
                    assert set(variables.keys()) == {'multicat'}
                    for variable_name, imputed in variables.items():
                        assert imputed.size == 1
                        assert imputed[0] in {0, 1, 2}

    def test_get_imputed_variables(self, categorical_imputer_fixture):
        train_imputed = categorical_imputer_fixture.get_imputed_variables(
            fold_name='train',
            split_i=0,
            imp_i=0
        )
        assert set(train_imputed.columns) == {'multicat'}
        train_unimputed, _ = categorical_imputer_fixture._split_then_join_Xy(0)
        train_unimputed = train_unimputed[['multicat']]

        assert train_unimputed.loc[
            train_unimputed.multicat.notnull(),
            'multicat'
        ].equals(
            train_imputed.loc[
                train_unimputed.multicat.notnull(),
                'multicat'
            ]
        )
        imputed = train_imputed.loc[
            train_unimputed.multicat.isnull(),
            'multicat'
        ].values
        assert imputed.size == 1
        assert imputed[0] in {0, 1, 2}

    def test_get_imputed_df(self, categorical_imputer_fixture):
        train_imputed = categorical_imputer_fixture.get_imputed_df(
            fold_name='train',
            split_i=0,
            imp_i=0
        )
        assert set(train_imputed.columns) == {
            'cont', 'bin', 'multicat', 'target'
        }
        train_unimputed, _ = categorical_imputer_fixture._split_then_join_Xy(0)
        assert train_imputed[['cont', 'target']].equals(
            train_unimputed[['cont', 'target']]
        )
        for var_name, possible_values in (
            ('bin', {0, 1}),
            ('multicat', {0, 1, 2})
        ):
            assert train_unimputed.loc[
                train_unimputed[var_name].notnull(),
                var_name
            ].equals(
                train_imputed.loc[
                    train_unimputed[var_name].notnull(),
                    var_name
                ]
            )
            imputed = train_imputed.loc[
                train_unimputed[var_name].isnull(),
                var_name
            ].values
            assert imputed.size == 1
            assert imputed[0] in possible_values
