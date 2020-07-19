import sys


import pytest
import pandas as pd
import numpy as np 

from utils.model.novel import NOVEL_MODEL_VARS

# @pytest.fixture
def dummy_data():
    """ generates fake data frame and makes it a pytest fixturr for use in tests"""
    df = pd.DataFrame()

    numrows = 1000
    #create continous columns
    for i in NOVEL_MODEL_VARS["cont"]:
        center = np.random.randint(50, 150)
        df[i] = np.random.normal(center, 10, numrows)

    for i in NOVEL_MODEL_VARS["cat"]:
        cats = (1.0, 2.0, 4.0, 8.0)
        df[i] = np.random.choice(cats, numrows)

    #introduce missingness
    for col in df.columns:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    df["target"] = np.random.randint(2, size=numrows)
    return(df)
        


dummy = dummy_data()
dummy2 = dummy.dropna()
print(dummy.shape, dummy2.shape)