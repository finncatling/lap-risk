import numpy as np
import pytest

from utils.model.novel import winsorize_novel, NOVEL_MODEL_VARS
from .fixtures import initial_df_fixture


def test_winsorise(initial_df_fixture):
    winsorised, thresholds = winsorize_novel(
        df=initial_df_fixture,
        thresholds=None,
        cont_vars=NOVEL_MODEL_VARS["cont"],
        quantiles=(0.2, 0.8)
    )

    # check the columns have changed
    with pytest.raises(AssertionError):
        for i in NOVEL_MODEL_VARS["cont"]:
            np.testing.assert_array_equal(
                initial_df_fixture[i].values,
                winsorised[i].values
            )
