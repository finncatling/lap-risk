import copy
from typing import Tuple, Dict, Callable
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from scipy.special import expit
from pygam import LinearGAM, s, f
from sklearn.preprocessing import QuantileTransformer

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
from utils.gam import quick_sample


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


@pytest.fixture()
def df_fixture_complete() -> pd.DataFrame:
    n_rows = 20
    rnd = np.random.RandomState(1)
    df = pd.DataFrame({'cont': np.linspace(-1, 1, num=n_rows)})
    df['bin'] = rnd.choice(2, size=n_rows)
    df['multicat'] = rnd.choice(3, size=n_rows)
    df['target'] = rnd.binomial(
        n=1, p=expit(df.cont + df.bin + (df.multicat - 1))
    )
    df['lacalb'] = ((df['cont'] * 2) + 0.3) ** 2
    return df


@pytest.fixture()
def df_fixture(df_fixture_complete) -> pd.DataFrame:
    df = df_fixture_complete.copy()
    df.loc[1, 'cont'] = np.nan
    df.loc[8, 'bin'] = np.nan
    df.loc[(3, 10), 'multicat'] = np.nan
    df.loc[(2, 17), 'lacalb'] = np.nan
    return df


@pytest.fixture()
def mock_train_test_splitter_fixture(df_fixture) -> TrainTestSplitter:
    """We use a mock to allow deterministic splitting"""
    tts = mock.Mock(TrainTestSplitter)
    even_i = np.arange(0, df_fixture.shape[0] - 1, 2)
    odd_i = np.arange(1, df_fixture.shape[0], 2)
    tts.train_i = [even_i, odd_i]
    tts.test_i = [odd_i, even_i]
    tts.n_splits = 2
    return tts


@pytest.fixture()
def splitter_winsor_mice_fixture(
    df_fixture,
    mock_train_test_splitter_fixture
) -> novel.SplitterWinsorMICE:
    swm = novel.SplitterWinsorMICE(
        df=df_fixture.drop(['multicat', 'lacalb'], axis=1),
        train_test_splitter=mock_train_test_splitter_fixture,
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


@pytest.fixture()
def categorical_imputer_fixture(
    df_fixture, splitter_winsor_mice_fixture
) -> novel.CategoricalImputer:
    cat_imputer = novel.CategoricalImputer(
        df=df_fixture.drop('lacalb', axis=1),
        splitter_winsor_mice=splitter_winsor_mice_fixture,
        cat_vars=['multicat'],
        random_seed=1
    )
    cat_imputer.impute()
    return cat_imputer


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


@pytest.fixture()
def lacalb_model_factory_fixture() -> Callable[
    [pd.Index, Dict[str, Tuple], str], LinearGAM
]:
    def model_factory(
        columns: pd.Index,
        multi_cat_levels: Dict[str, Tuple],
        indication_var_name: str
    ) -> LinearGAM:
        return LinearGAM(
            s(columns.get_loc("cont"), spline_order=2, n_splines=5, lam=0.05)
            + f(columns.get_loc("bin"), coding="dummy", lam=1000)
            + f(columns.get_loc("multicat"), coding="dummy", lam=1000)
        )
    return model_factory


@pytest.fixture()
def lacalb_imputer_fixture(
    df_fixture,
    categorical_imputer_fixture,
    lacalb_model_factory_fixture
) -> novel.LactateAlbuminImputer:
    imp = novel.LactateAlbuminImputer(
        df=df_fixture.drop(['cont', 'bin', 'multicat'], axis=1),
        categorical_imputer=categorical_imputer_fixture,
        lacalb_variable_name='lacalb',
        imputation_model_factory=lacalb_model_factory_fixture,
        winsor_quantiles=novel.WINSOR_QUANTILES,
        multi_cat_vars=dict(),  # unused
        indication_var_name='',  # unused
        random_seed=1
    )
    imp.fit()
    return imp


class TestLactateAlbuminImputer:
    """
    _split() and _find_missing_indices() are base class methods tested
    elsewhere, so aren't retested here.
    """
    def test_check_df(
        self,
        df_fixture,
        categorical_imputer_fixture,
        lacalb_model_factory_fixture
    ):
        """Should raise AssertionError as too many columns in df."""
        with pytest.raises(AssertionError):
            return novel.LactateAlbuminImputer(
                df=df_fixture,
                categorical_imputer=categorical_imputer_fixture,
                lacalb_variable_name='lacalb',
                imputation_model_factory=lacalb_model_factory_fixture,
                winsor_quantiles=novel.WINSOR_QUANTILES,
                multi_cat_vars=dict(),  # unused
                indication_var_name='',  # unused
                random_seed=1
            )

    @pytest.fixture()
    def obs_lacalb_train_fixture(self, lacalb_imputer_fixture) -> pd.DataFrame:
        lacalb_train, _, _, _ = lacalb_imputer_fixture._split(0)
        return lacalb_imputer_fixture._get_observed_values(
            fold="train",
            split_i=0,
            X=lacalb_train)

    @pytest.fixture()
    def expected_obs_lacalb_train_fixture(self, df_fixture) -> pd.DataFrame:
        even_i = np.arange(0, df_fixture.shape[0] - 1, 2)
        df = df_fixture.loc[even_i, ['lacalb']].reset_index(drop=True)
        return df.loc[df['lacalb'].notnull()]

    def test_get_observed_values(
        self, obs_lacalb_train_fixture, expected_obs_lacalb_train_fixture
    ):
        assert obs_lacalb_train_fixture.equals(
            expected_obs_lacalb_train_fixture)

    @pytest.fixture()
    def obs_lacalb_train_winsorized_fixture(
        self, lacalb_imputer_fixture, obs_lacalb_train_fixture
    ) -> pd.DataFrame:
        return lacalb_imputer_fixture._winsorize(
            split_i=0,
            lacalb=obs_lacalb_train_fixture)

    @pytest.fixture()
    def expected_obs_lacalb_train_winsorized_fixture(
        self, expected_obs_lacalb_train_fixture
    ) -> pd.DataFrame:
        df = expected_obs_lacalb_train_fixture
        thresholds = list(df['lacalb'].quantile(novel.WINSOR_QUANTILES))
        df.loc[df['lacalb'] < thresholds[0], 'lacalb'] = thresholds[0]
        df.loc[df['lacalb'] > thresholds[1], 'lacalb'] = thresholds[1]
        return df

    def test_winsorize(
        self,
        obs_lacalb_train_winsorized_fixture,
        expected_obs_lacalb_train_winsorized_fixture
    ):
        assert obs_lacalb_train_winsorized_fixture.equals(
            expected_obs_lacalb_train_winsorized_fixture)

    @pytest.fixture()
    def obs_lacalb_train_trans_fixture(
        self, lacalb_imputer_fixture, obs_lacalb_train_winsorized_fixture
    ) -> pd.DataFrame:
        return lacalb_imputer_fixture._fit_transform(
            split_i=0,
            obs_lacalb_train=obs_lacalb_train_winsorized_fixture)

    def test_fit_transform(
        self,
        lacalb_imputer_fixture,
        obs_lacalb_train_trans_fixture
    ):
        """QuantileTransformer should transform to unit Gaussian, but variance
            is nowhere near 1 here, likely because the dataset is so small."""
        assert len(lacalb_imputer_fixture.transformers) == 2
        for trans in lacalb_imputer_fixture.transformers.values():
            assert isinstance(trans, QuantileTransformer)
        df = obs_lacalb_train_trans_fixture
        assert df['lacalb'].mean() < 0.1 and df['lacalb'].mean() > -0.1
        # assert df['lacalb'].var() < 1.1 and df['lacalb'].var() > 0.9

    def test_fit_combine_gams(self, lacalb_imputer_fixture):
        assert len(lacalb_imputer_fixture.imputers) == 2
        for imp in lacalb_imputer_fixture.imputers.values():
            assert isinstance(imp, LinearGAM)
            assert len(imp.terms._terms) == 4  # cont, bin, multicat & intercept
        # import matplotlib.pyplot as plt
        # import os
        # from utils.constants import INTERNAL_OUTPUT_DIR
        # plt.figure()
        # fig, axs = plt.subplots(1, 3, figsize=(8, 2))
        # axs.ravel()
        # imp = lacalb_imputer_fixture.imputers[0]
        # for j, ax in enumerate(axs):
        #     XX = imp.generate_X_grid(term=j)
        #     ax.plot(XX[:, j], imp.partial_dependence(term=j, X=XX))
        #     ax.plot(XX[:, j],
        #             imp.partial_dependence(term=j, X=XX, width=.95)[1],
        #             c='r', ls='--')
        # fig.tight_layout()
        # fig.savefig(
        #     os.path.join(INTERNAL_OUTPUT_DIR, f"lacalb_gam.pdf"),
        #     format='pdf',
        #     bbox_inches="tight")

    @pytest.fixture()
    def missing_features_fixture(self, lacalb_imputer_fixture) -> pd.DataFrame:
        return lacalb_imputer_fixture._get_features_where_lacalb_missing(
            fold_name='train',
            split_i=0,
            mice_imp_i=0)

    def test_get_features_where_lacalb_missing(
        self, lacalb_imputer_fixture, df_fixture, missing_features_fixture
    ):
        """expected is only so simple to construct here because cont's
        value isn't at an extreme of its domain (so wouldn't be winsorized
        anyway), and none of the features has a missing values which would
        need to be imputed."""
        even_i = np.arange(0, df_fixture.shape[0] - 1, 2)
        df = df_fixture.loc[even_i].reset_index(drop=True)
        expected = df.loc[
            df['lacalb'].isnull()
        ].drop(['target', 'lacalb'], axis=1)
        assert expected.equals(missing_features_fixture)

    def test_impute_non_probabilistic(
        self,
        lacalb_imputer_fixture,
        missing_features_fixture,
        df_fixture_complete
    ):
        pred = lacalb_imputer_fixture.impute(
            features=missing_features_fixture,
            split_i=0,
            lac_alb_imp_i=None,
            probabilistic=False)[0][0]
        true = df_fixture_complete.loc[2, 'lacalb']
        assert pred > (true - 0.5)
        assert pred < (true + 0.5)

    def test_get_complete_lacalb(
        self,
        lacalb_imputer_fixture,
        missing_features_fixture,
        df_fixture
    ):
        pred = lacalb_imputer_fixture.impute(
            features=missing_features_fixture,
            split_i=0,
            lac_alb_imp_i=None,
            probabilistic=False)
        observed = lacalb_imputer_fixture._get_complete_lacalb(pred, 'train', 0)
        even_i = np.arange(0, df_fixture.shape[0] - 1, 2)
        df = df_fixture.loc[even_i].reset_index(drop=True)
        df.loc[1, 'lacalb'] = pred[0][0]
        assert df.loc[:, ['lacalb']].equals(observed)
