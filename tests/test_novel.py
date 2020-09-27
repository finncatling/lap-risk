import pytest

from .datafixture import dummy_df
from utils.model import novel 

df = dummy_df()

def test_combine():
    mapping = {"S03ECG": {
        1 : 1,
        2 : 2,
        4 : 4,
        8:  4,
    }}
    combined_df = novel.combine_categories(df, mapping)

    assert df["S03ECG"].unique != combined_df["S03ECG"].unique

def test_add_missingness():
    cols = df.shape[1]
    df2 = novel.add_missingness_indicators(df, ("S01AgeOnArrival", "S03SerumCreatinine"))
    cols = df2.shape[1] - cols
    assert cols == 2

