import pandas as pd
from utils.model import current


def test_discretise_gcs():
    df = pd.DataFrame({"S03GlasgowComaScore": [3., 8., 9., 12., 13., 14]})
    df = current.discretise_gcs(df)
    assert pd.DataFrame({
        'gcs_3_8': [1., 1., 0., 0., 0., 0.],
        'gcs_9_12': [0., 0., 1., 1., 0., 0.]
    }).equals(df)


def test_binarize_categorical():
    df = pd.DataFrame({
        'binary': [5., 6., 5.],
        'multi_cat': [2., 3., 4.],
        'ignore': [1.5, 2.3, 9.1]
    })
    df, _ = current.binarize_categorical(
        df=df,
        label_binarizers=None,
        binarize_vars=['binary', 'multi_cat']
    )
    assert all(pd.DataFrame({
        'binary': [0, 1, 0],
        'ignore': [1.5, 2.3, 9.1],
        'multi_cat_2': [1, 0, 0],
        'multi_cat_3': [0, 1, 0],
        'multi_cat_4': [0, 0, 1]
    }) == df)


def test_winsorize_current():
    df = pd.DataFrame({
        'for_winsor': [6., 1., 20.],
        'ignore': [0., 1., 0.]
    })
    df = current.winsorize_current(
        df=df,
        winsor_thresholds={'for_winsor': (4., 10.)}
    )
    assert pd.DataFrame({
        'for_winsor': [6., 4., 10.],
        'ignore': [0., 1., 0.]
    }).equals(df)


def test_add_asa_age_resp_interaction():
    assert False


def test_transform_sodium():
    assert False


class TestSplitterTrainerPredictor:
    def test_split_train_predict(self):
        assert False

    def test__train(self):
        assert False
