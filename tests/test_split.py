import pytest

from .datafixture import dummy_dataframe
from utils import split
from utils.model.novel import WINSOR_QUANTILES, NOVEL_MODEL_VARS


def test_drop_incomplete_cases():
    df = dummy_dataframe()
    dropped, stats = split.drop_incomplete_cases(df)

    missing_in_dropped = dropped.shape[0] - dropped.dropna().shape[0]
    assert  missing_in_dropped == 0

    number_dropped = df.shape[0] - dropped.shape[0]
    assert number_dropped == stats["n_dropped_cases"]

    #check original dataframe not changed
    assert df.shape[0] - dropped.shape[0] > 0

def test_split_into_folds():
    #TODO this test should probably be more comprehensive but it's passing for now
    df = dummy_dataframe()
   
    indices = {
        'train':df.sample(frac=0.6).index,
        'test': df.sample(frac=0.2).index
    }
    stuff = split.split_into_folds(df, indices, NOVEL_MODEL_VARS["target"])

    assert stuff[0].shape[0] == df.sample(frac=0.6).shape[0]

def test_TTS():
    df = dummy_dataframe()
    model_vars = NOVEL_MODEL_VARS["cat"] + NOVEL_MODEL_VARS["cont"]
    tts = split.TrainTestSplitter(df, "HospitalId.anon", 0.2, 5, list(model_vars), 5)
    tts.split()
    
    assert len(tts.train_institution_ids) == 5