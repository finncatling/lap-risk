from typing import List

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_absolute_error

from utils import evaluate
from utils.simulate import simulate_labels_and_well_calibrated_pred_probs


class TestModelScorer:
    """End-to-end test which just checks that"""

    @pytest.fixture(scope='class')
    def ms_fixture(self) -> evaluate.ModelScorer:
        y_true, y_pred = [], []
        for i in range(4):
            true, pred = simulate_labels_and_well_calibrated_pred_probs(
                n_labels=100,
                random_seed=i
            )
            y_true.append(true)
            y_pred.append(pred)
        ms = evaluate.ModelScorer(
            y_true=y_true,
            y_pred=y_pred,
            scorer_function=evaluate.score_logistic_predictions,
            calibration_n_splines=5,
            calibration_lam_candidates=np.logspace(-3, -1, 5)
        )
        ms.calculate_scores()
        return ms

    def test_n_splits(self, ms_fixture):
        assert ms_fixture.n_splits == 4

    @pytest.fixture(scope='class')
    def score_names(self, ms_fixture) -> List[str]:
        return list(ms_fixture.scores['per_split'][0])

    def test_score_names(self, score_names):
        assert len(score_names) > 0
        assert all([isinstance(name, str) for name in score_names])

    def test_scores_dict_format(self, ms_fixture, score_names):
        for subdict_name in ['per_score', 'per_score_diff', '95ci']:
            assert score_names == list(ms_fixture.scores[subdict_name])
        for score_name in score_names:
            assert len(ms_fixture.scores['95ci'][score_name]) == 2
            for subdict_name in ['per_score', 'per_score_diff']:
                assert (
                    len(ms_fixture.scores[subdict_name][score_name]) ==
                    ms_fixture.n_splits
                )

    def test_95ci(self, ms_fixture, score_names):
        for score_name in score_names:
            assert (
                ms_fixture.scores['95ci'][score_name][1] <
                ms_fixture.scores['per_score'][score_name].max()
            )
            assert (
                ms_fixture.scores['95ci'][score_name][0] >
                ms_fixture.scores['per_score'][score_name].min()
            )


def test_score_calibration():
    calibration_error_threshold = 0.05
    y_true, y_pred = simulate_labels_and_well_calibrated_pred_probs(
        n_labels=1000,
        random_seed=1
    )

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


def test_somers_dxy():
    assert evaluate.somers_dxy(
        y_true=np.array([1., 0.]),
        y_pred=np.array([0.75, 0.25])
    ) == 1.0
    assert evaluate.somers_dxy(
        y_true=np.array([1., 1., 0., 0.]),
        y_pred=np.array([0.75, 0.25, 0.75, 0.25])
    ) == 0.0
    assert evaluate.somers_dxy(
        y_true=np.array([1., 1., 0., 0.]),
        y_pred=np.array([0.75, 0.75, 0.75, 0.25])
    ) == 0.5


def test_tjurs_coef():
    assert evaluate.tjurs_coef(
        y_true=np.array([1., 0.]),
        y_pred=np.array([0.65, 0.45])
    ) == 0.2
    assert evaluate.tjurs_coef(
        y_true=np.array([1., 1., 0., 0.]),
        y_pred=np.array([0.75, 0.25, 0.75, 0.25])
    ) == 0.0


def test_stratify_y_pred():
    y_pred_0, y_pred_1 = evaluate.stratify_y_pred(
        y_true=np.array([0., 1., 0., 1.]),
        y_pred=np.array([0.25, 0.75, 0.35, 0.65])
    )
    assert (y_pred_0 == np.array([0.25, 0.35])).all()
    assert (y_pred_1 == np.array([0.75, 0.65])).all()
