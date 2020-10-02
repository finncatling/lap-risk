import pandas as pd
from utils.model import current


def test_discretise_gcs():
    df = pd.DataFrame({"S03GlasgowComaScore": [3., 8., 9., 12., 13., 14]})
    df = current.discretise_gcs(df)
    assert pd.DataFrame({
        'gcs_3_8': [1., 1., 0., 0., 0., 0.],
        'gcs_9_12': [0., 0., 1., 1., 0., 0.]
    }).equals(df)
