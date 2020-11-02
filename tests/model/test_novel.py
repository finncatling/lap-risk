import copy
from typing import Tuple, Dict
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from scipy.special import expit

import utils.model
from utils.indications import (
    INDICATION_VAR_NAME,
    MISSING_IND_CATEGORY,
    get_indication_variable_names
)
from utils.model import novel
from utils.model.novel import (
    NOVEL_MODEL_VARS,
    MULTI_CATEGORY_LEVELS,
    preprocess_novel_pre_split
)
from utils.model.shared import flatten_model_var_dict
from utils.split import TrainTestSplitter


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


def test_label_encode():
    le_df = novel.label_encode(
        df=pd.DataFrame({
            'a': [np.nan, 3., 2., 1.],
            'Indication': ['1', '0', '4', '0']
        }),
        multi_cat_levels={
            'a': (1., 2., 3.),
            'Indication': ('0', '4')
        },
        missing_indication_value='1'
    )
    assert pd.DataFrame({
        'a': [np.nan, 2., 1., 0.],
        'Indication': [np.nan, 0., 1., 0.]
    }).equals(le_df)


def test_preprocess_novel_pre_split(initial_df_permutations_fixture):
    """End-to-end test which just checks that dtypes and column names are as
        expected."""
    df = initial_df_permutations_fixture

    indications = get_indication_variable_names(df.columns)
    df = df[flatten_model_var_dict(NOVEL_MODEL_VARS) + indications]

    multi_category_levels: Dict[str, Tuple] = copy.deepcopy(
        MULTI_CATEGORY_LEVELS)
    multi_category_levels[INDICATION_VAR_NAME] = tuple(indications)

    df = preprocess_novel_pre_split(
        df,
        category_mapping={"S03ECG": {1.0: 0.0, 4.0: 1.0, 8.0: 1.0}},
        indication_variable_name=INDICATION_VAR_NAME,
        indications=indications,
        missing_indication_value=MISSING_IND_CATEGORY,
        multi_category_levels=multi_category_levels,
    )

    for column_name in df.columns:
        assert is_numeric_dtype(df[column_name])

    assert set(df.columns) == {
        "S03ASAScore",
        "S03CardiacSigns",
        "S03RespiratorySigns",
        "S03DiagnosedMalignancy",
        "S03Pred_Peritsoil",
        "S02PreOpCTPerformed",
        "S03ECG",
        "S01AgeOnArrival",
        "S03SerumCreatinine",
        "S03PreOpArterialBloodLactate",
        "S03PreOpLowestAlbumin",
        "S03Sodium",
        "S03Potassium",
        "S03Urea",
        "S03WhiteCellCount",
        "S03Pulse",
        "S03SystolicBloodPressure",
        "S03GlasgowComaScore",
        "Target",
        "Indication"
    }


class TestWinsorizeNovel:
    @pytest.fixture()
    def input_df_fixture(self):
        return pd.DataFrame({
            'a': [0.0, 0.25, 0.5, 0.75, 1.0],
            'b': [1.0, 1.25, 1.5, 1.75, 2.0],
            'ignore': [0, 1, 0, 1, 0]
        })

    @pytest.fixture()
    def thresholds_fixture(self):
        return {'a': [0.2, 0.8], 'b': [None, 1.8]}

    @pytest.fixture()
    def output_df_fixture(self):
        return pd.DataFrame({
            'a': [0.2, 0.25, 0.5, 0.75, 0.8],
            'b': [1.0, 1.25, 1.5, 1.75, 1.8],
            'ignore': [0, 1, 0, 1, 0]
        })

    def test_winsorise_novel_quantiles_input(
        self, input_df_fixture, output_df_fixture, thresholds_fixture
    ):
        winsor_df, thresholds = novel.winsorize_novel(
            df=input_df_fixture,
            cont_vars=['a', 'b'],
            quantiles=(0.2, 0.8),
            include={'b': (False, True)}
        )
        assert thresholds == thresholds_fixture
        assert output_df_fixture.equals(winsor_df)

    def test_winsorise_novel_thresholds_input(
        self, input_df_fixture, output_df_fixture, thresholds_fixture
    ):
        winsor_df, thresholds = novel.winsorize_novel(
            df=input_df_fixture,
            thresholds=thresholds_fixture
        )
        assert thresholds == thresholds_fixture
        assert output_df_fixture.equals(winsor_df)


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
    tts.n_iters = len(tts.train_i)
    return tts


@pytest.fixture(scope='module')
def splitter_winsor_mice_fixture(
    df_fixture, mock_train_test_split_fixture
) -> utils.model.novel.SplitterWinsorMICE:
    swm = utils.model.novel.SplitterWinsorMICE(
        df=df_fixture.drop('multicat', axis=1),
        train_test_splitter=mock_train_test_split_fixture,
        target_variable_name='target',
        cont_variables=['cont'],
        binary_variables=['bin'],
        winsor_quantiles=novel.WINSOR_QUANTILES,
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
) -> utils.model.novel.CategoricalImputer:
    cat_imputer = utils.model.novel.CategoricalImputer(
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
        train_unimputed, _ = novel.winsorize_novel(
            df=train_unimputed,
            quantiles=novel.WINSOR_QUANTILES,
            cont_vars=['cont']
        )
        print(train_imputed)
        print(train_unimputed)
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


class TestLactateAlbuminImputer:
    def test_placeholder(self):
        assert False
