from typing import Tuple

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_absolute_error

from utils import evaluate


class TestModelScorer:
    def test_n_splits(self):
        assert False

    def test__sanity_check(self):
        assert False

    def test_calculate_scores(self):
        assert False

    def test_print_scores(self):
        assert False

    def test__calculate_95ci(self):
        assert False

    def test__extract_iter_per_score(self):
        assert False


@pytest.fixture
def labels_and_well_calibrated_pred_probs() -> Tuple[np.ndarray, np.ndarray]:
    """Simulates binary labels and corresponding well-calibrated predicted
        probabilities for the positive class

    Returns:
        y_true
        y_predicted_probabilities
    """
    rnd = np.random.RandomState(1)
    y_pred = rnd.uniform(low=0.0, high=1.0, size=1000)
    y_true = rnd.binomial(n=1, p=y_pred)
    return y_true, y_pred


def test_score_calibration(labels_and_well_calibrated_pred_probs):
    calibration_error_threshold = 0.03
    y_true, y_pred = labels_and_well_calibrated_pred_probs

    # Test that we report low calibration error with these perfect predictions
    _, our_cal_curve, our_calib_mae, _ = evaluate.score_calibration(
        y_true=y_true,
        y_pred=y_pred,
        n_splines=5,
        lam_candidates=np.logspace(-3, -1, 5)
    )
    assert our_calib_mae < calibration_error_threshold

    # Test that our calibration curve is similar to sklearn's
    sklearn_calibration_curve, sklearn_prob_pred = calibration_curve(
        y_true,
        y_pred,
        n_bins=5
    )
    downsample_i = np.round(sklearn_prob_pred * 100).astype(int)
    our_cal_curve_downsampled = np.array(
        [our_cal_curve[i] for i in downsample_i]
    )
    difference_from_sklearn = mean_absolute_error(
        sklearn_calibration_curve,
        our_cal_curve_downsampled
    )
    assert difference_from_sklearn < calibration_error_threshold
