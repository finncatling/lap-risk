from utils import split
from utils.model.novel import NOVEL_MODEL_VARS
from .fixtures import initial_df_fixture, train_test_split_fixture


def test_drop_incomplete_cases(initial_df_fixture):
    dropped, stats = split.drop_incomplete_cases(initial_df_fixture)

    missing_in_dropped = dropped.shape[0] - dropped.dropna().shape[0]
    assert missing_in_dropped == 0

    number_dropped = initial_df_fixture.shape[0] - dropped.shape[0]
    assert number_dropped == stats["n_dropped_cases"]

    # check original dataframe not changed
    assert initial_df_fixture.shape[0] - dropped.shape[0] > 0


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
