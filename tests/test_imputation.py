import pytest

from .datafixture import dummy_df, TTS
from utils import impute
from utils.split import Splitter, TrainTestSplitter
from utils.model.novel import WINSOR_QUANTILES, NOVEL_MODEL_VARS

df = dummy_df()
#includes hospital.id which needs to be dropped
df = df.drop(columns=["HospitalId.anon"])


def test_determine_imputations():
    n_imps, fraction = impute.determine_n_imputations(df)
    
    assert n_imps == 59

def test_SWM_1():
    swm = impute.SplitterWinsorMICE(
    df=df,
    train_test_splitter=TTS(),
    target_variable_name=NOVEL_MODEL_VARS["target"],
    cont_variables=NOVEL_MODEL_VARS["cont"],
    binary_variables=NOVEL_MODEL_VARS["cat"],
    winsor_quantiles=(0.2, 0.8),
    winsor_include={
        "S01AgeOnArrival": (False, True),
        "S03GlasgowComaScore": (False, False),
    },
    n_mice_imputations=5,
    n_mice_burn_in=10,
    n_mice_skip=3,
    )

    swm.split_winsorize_mice()
    breakpoint()