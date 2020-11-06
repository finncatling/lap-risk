from typing import Tuple
import numpy as np
import pandas as pd
from pygam import LinearGAM, LogisticGAM, s
from scipy.special import expit
from utils.gam import quick_sample


def lineargam_data(n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Data requiring a polynomial / spline fit."""
    df = pd.DataFrame({'x': np.linspace(-1, 1, num=n_rows)})
    df['y'] = ((df['x'] * 2) + 0.3) ** 2
    return df[['x']].values, df.y.values


def test_quick_sample_lineargam_y():
    """Check concordance between the mean of the samples obtained with
    quick_sample() and the predictions from pygam's inbuilt GAM.predict()"""
    X, y = lineargam_data(n_rows=20)
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

    y_samples_mean = np.mean(y_samples, axis=0)
    assert all(np.abs(y_pred - y_samples_mean) < 0.1)


def logisticgam_data(n_rows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({'x': np.linspace(0, 2 * np.pi, num=n_rows)})
    df['p_y'] = expit(np.sin(df.x) * 2)
    rnd = np.random.RandomState(1)
    df['y'] = rnd.binomial(n=1, p=df.p_y.values)
    return df[['x']].values, df.y.values, df.p_y.values


def test_quick_sample_logisticgam_mu():
    """Check concordance between the predicted probability samples obtained
    with quick_sample() and the ground truth probability."""
    X, y, p_y = logisticgam_data(n_rows=100)
    gam = LogisticGAM(
        s(0, spline_order=2, n_splines=5, lam=0.001),
        fit_intercept=False)
    gam.fit(X, y)

    y_samples = quick_sample(
        gam=gam,
        sample_at_X=X,
        quantity='mu',
        n_draws=100,
        random_seed=0)

    y_samples_mean = np.mean(y_samples, axis=0)
    assert all(np.abs(p_y - y_samples_mean) < 0.25)
