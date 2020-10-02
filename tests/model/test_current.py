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
    """Patient in row 0 fell into the (now dropped) base category 1.0 for
        respiratory signs, hence initial 0 in the S03RespiratorySigns_2 and
        03RespiratorySigns_4 columns."""
    df = pd.DataFrame({
        "S01AgeOnArrival": [1., 2., 3., 4., 5.],
        "S01AgeOnArrival_2": [1., 4., 9., 16., 25.],
        "S03RespiratorySigns_2": [0, 1, 1, 0, 0],
        "S03RespiratorySigns_4": [0, 0, 0, 1, 1],
        "S03ASAScore": [1., 2., 3., 4., 5.]
    })
    df = current.add_asa_age_resp_interaction(df)
    assert pd.DataFrame({
        "age_asa12": [1., 2., 0., 0., 0.],
        "age_2_asa12": [1., 4., 0., 0., 0.],
        "resp2_asa12": [0., 1., 0., 0., 0.],
        "resp4_asa12": [0., 0., 0., 0., 0.],
        "age_asa3": [0., 0., 3., 0., 0.],
        "age_2_asa3": [0., 0., 9., 0., 0.],
        "resp2_asa3": [0., 0., 1., 0., 0.],
        "resp4_asa3": [0., 0., 0., 0., 0.],
        "age_asa4": [0., 0., 0., 4., 0.],
        "age_2_asa4": [0., 0., 0., 16., 0.],
        "resp2_asa4": [0., 0., 0., 0., 0.],
        "resp4_asa4": [0., 0., 0., 1., 0.],
        "age_asa5": [0., 0., 0., 0., 5.],
        "age_2_asa5": [0., 0., 0., 0., 25.],
        "resp2_asa5": [0., 0., 0., 0., 0.],
        "resp4_asa5": [0., 0., 0., 0., 1.]
    }).equals(df)


def test_preprocess_current():
    """End-to-end test which just checks that column names are as expected."""
    assert False


class TestSplitterTrainerPredictor:
    def test_split_train_predict(self):
        assert False

    def test__train(self):
        assert False
