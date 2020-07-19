import pytest
import numpy as np
from .datafixture import dummy_dataframe
from utils.model import novel

df = dummy_dataframe()

def test_winsorise_1():
    winsorised, thresholds = novel.winsorize_novel(df, None, novel.NOVEL_MODEL_VARS["cont"], (0.2, 0.8))
    with pytest.raises(AssertionError):
        for i in novel.NOVEL_MODEL_VARS["cont"]:
            np.testing.assert_array_equal(df[i].values, winsorised[i].values)

def test_winsorise_2():
    winsorised, thresholds = novel.winsorize_novel(df, None, novel.NOVEL_MODEL_VARS["cont"], (0.2, 0.8))
    with pytest.raises(AssertionError):
        for i in novel.NOVEL_MODEL_VARS["cont"]:
            np.testing.assert_array_equal(df[i].values, winsorised[i].values)