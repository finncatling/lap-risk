import copy
from typing import List, Dict

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_absolute_error
from unittest.mock import Mock

from utils import evaluate
from utils.simulate import simulate_labels_and_well_calibrated_pred_probs


class TestModelScorer:
    """End-to-end test."""

    @pytest.fixture(scope='class')
    def ms_fixture(self) -> evaluate.LogisticScorer:
        y_true, y_pred = [], []
        for i in range(4):
            true, pred = simulate_labels_and_well_calibrated_pred_probs(
                n_labels=100,
                random_seed=i
            )
            y_true.append(true)
            y_pred.append(pred)
        ms = evaluate.LogisticScorer(
            y_true=y_true,
            y_pred=y_pred,
            scorer_function=evaluate.score_logistic_predictions,
            n_splits=2,
            calibration_n_splines=5,
            calibration_lam_candidates=np.logspace(-3, -1, 5)
        )
        ms.calculate_scores()
        return ms

    def test_n_iters(self, ms_fixture):
        assert ms_fixture.n_iters == 4

    def test_n_iters_per_split(self, ms_fixture):
        assert ms_fixture.n_iters_per_split == 2

    @pytest.fixture(scope='class')
    def score_names(self, ms_fixture) -> List[str]:
        return list(ms_fixture.scores['per_iter'][0])

    def test_score_names(self, score_names):
        assert len(score_names) > 0
        assert all([isinstance(name, str) for name in score_names])

    def test_scores_dict_format(self, ms_fixture, score_names):
        for subdict_name in ['per_score', 'per_score_diff', '95ci', 'medians']:
            assert score_names == list(ms_fixture.scores[subdict_name])
        for score_name in score_names:
            assert len(ms_fixture.scores['95ci'][score_name]) == 2
            for subdict_name in ['per_score', 'per_score_diff']:
                assert (
                    len(ms_fixture.scores[subdict_name][score_name]) ==
                    ms_fixture.n_iters
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

    def test_medians(self, ms_fixture, score_names):
        for score_name in score_names:
            assert (
                ms_fixture.scores['95ci'][score_name][1] >
                ms_fixture.scores['medians'][score_name]
            )
            assert (
                ms_fixture.scores['95ci'][score_name][0] <
                ms_fixture.scores['medians'][score_name]
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


class TestScoreComparer:
    @pytest.fixture()
    def scorer_fixture(self) -> evaluate.Scorer:
        scorer = evaluate.Scorer(
            y_true=[np.array([4.6, 1.8, 7.2]), np.array([3.2, 0.3, 9.1])],
            y_pred=[np.array([4.1, 2.1, 7.6]), np.array([3.0, 0.6, 8.8])],
            scorer_function=evaluate.score_linear_predictions,
            n_splits=2
        )
        scorer.calculate_scores()
        return scorer

    @pytest.fixture()
    def y_true_labels(self) -> List[np.ndarray]:
        return [np.array([1, 0, 0]), np.array([0, 0, 1])]

    @pytest.fixture()
    def logistic_scorer_args(self) -> Dict:
        return {
            'y_true': [np.array([1, 0, 0]), np.array([0, 0, 1])],
            'scorer_function': evaluate.score_logistic_predictions,
            'n_splits': 2,
            'calibration_n_splines': 5,
            'calibration_lam_candidates': np.array([0.1, 1])}

    @pytest.fixture()
    def logistic_scorer_1_fixture(
        self, logistic_scorer_args
    ) -> evaluate.LogisticScorer:
        logistic_scorer = evaluate.LogisticScorer(
            y_pred=[np.array([0.6, 0.4, 0.6]), np.array([0.4, 0.5, 0.6])],
            **logistic_scorer_args)
        logistic_scorer.calculate_scores()
        return logistic_scorer

    @pytest.fixture()
    def logistic_scorer_2_fixture(
        self, logistic_scorer_args
    ) -> evaluate.LogisticScorer:
        logistic_scorer = evaluate.LogisticScorer(
            y_pred=[np.array([0.9, 0.1, 0.2]), np.array([0.1, 0.2, 0.8])],
            **logistic_scorer_args)
        logistic_scorer.calculate_scores()
        return logistic_scorer

    @pytest.fixture()
    def score_comparer_fixture(
        self, logistic_scorer_1_fixture, logistic_scorer_2_fixture
    ) -> evaluate.ScoreComparer:
        sc = evaluate.ScoreComparer(
            scorers=(logistic_scorer_1_fixture, logistic_scorer_2_fixture),
            scorer_names=('1', '2'))
        sc.compare_scores()
        return sc

    def test_score_names(self, score_comparer_fixture):
        assert len(score_comparer_fixture.score_names) > 0
        assert all([isinstance(name, str) for name in
                    score_comparer_fixture.score_names])

    def test_sanity_check_types(
        self, scorer_fixture, logistic_scorer_1_fixture):
        with pytest.raises(AssertionError):
            evaluate.ScoreComparer(
                scorers=(scorer_fixture, logistic_scorer_1_fixture),
                scorer_names=('1', '2'))

    def test_sanity_check_lengths(self, scorer_fixture):
        scorer_2 = copy.deepcopy(scorer_fixture)
        scorer_3 = copy.deepcopy(scorer_fixture)
        with pytest.raises(AssertionError):
            evaluate.ScoreComparer(
                scorers=(scorer_fixture, scorer_2, scorer_3),
                scorer_names=('1', '2'))
        with pytest.raises(AssertionError):
            evaluate.ScoreComparer(
                scorers=(scorer_fixture, scorer_2),
                scorer_names=('1',))

    def test_sanity_check_score_names(
        self, logistic_scorer_1_fixture, logistic_scorer_2_fixture):
        logistic_scorer_1_fixture.scores["per_iter"][0]['another score'] = 0.0
        with pytest.raises(AssertionError):
            evaluate.ScoreComparer(
                scorers=(logistic_scorer_1_fixture, logistic_scorer_2_fixture),
                scorer_names=('1', '2'))

    def test_score_diff_dict_format(self, score_comparer_fixture):
        diff_dict = score_comparer_fixture.score_diff
        score_names = score_comparer_fixture.score_names
        for subdict_name in ['splits', 'split0', '95ci', 'median']:
            assert score_names == list(diff_dict[subdict_name])
        for score_name in score_names:
            assert len(diff_dict['95ci'][score_name]) == 2
            assert (
                len(diff_dict['splits'][score_name]) ==
                len(score_comparer_fixture.scorers[0].y_true)
            )

    def test_95ci(self, score_comparer_fixture):
        for score_name in score_comparer_fixture.score_names:
            assert (
                score_comparer_fixture.score_diff['95ci'][score_name][1] <
                score_comparer_fixture.score_diff['splits'][score_name].max()
            )
            assert (
                score_comparer_fixture.score_diff['95ci'][score_name][0] >
                score_comparer_fixture.score_diff['splits'][score_name].min()
            )

    def test_medians(self, score_comparer_fixture):
        for score_name in score_comparer_fixture.score_names:
            assert (
                score_comparer_fixture.score_diff['95ci'][score_name][1] >
                score_comparer_fixture.score_diff['median'][score_name]
            )
            assert (
                score_comparer_fixture.score_diff['95ci'][score_name][0] <
                score_comparer_fixture.score_diff['median'][score_name]
            )
