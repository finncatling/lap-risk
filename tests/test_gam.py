from typing import Tuple
import pytest
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from utils.gam import quick_sample


@pytest.fixture()
def gam_data() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({'x': np.linspace(-1, 1, num=20)})
    df['y'] = ((df['x'] * 2) + 0.3) ** 2
    return df[['x']].values, df.y.values


def test_quick_sample_y_quantity(gam_data):
    X, y = gam_data
    gam = LinearGAM(
        s(0, spline_order=2, n_splines=5, lam=0.1),
        fit_intercept=False)
    gam.fit(X, y)

    y_pred = gam.predict(X)
    y_samples = quick_sample(
        gam=gam,
        sample_at_X=X,
        quantity='y',
        n_draws=200,
        random_seed=0)
    y_sample_mean = np.mean(y_samples, axis=0)

    diff = np.abs(y_pred - y_sample_mean)
    assert all(diff < 0.1)
