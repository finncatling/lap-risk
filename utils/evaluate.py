from typing import List, Dict, Union, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from progressbar import progressbar as pb
from pygam import GAM
from pygam.distributions import BinomialDist
from pygam.terms import s
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    median_absolute_error
)


def score_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_splines: int,
    lam_candidates: np.ndarray
) -> (np.ndarray, np.ndarray, float, float):
    """Given binary labels and corresponding predicted probabilities from a
        binary logistic model, derives a smooth calibration curve for that
        binary logistic model. Reports calibration error versus line of
        identity.

        calib_gam models y_true as multivariate Bernoulli-distributed
        (equivalent to multivariate binomial distribution where trials=1). This
        distribution is parameterised by sigmoid(f(y_pred)) and f() is a
        learned spline transformation. f() transforms y_pred from [0, 1] to
        [-inf, inf], and sigmoid() transforms f(y_pred) back to [0, 1].

        For each mortality label y_true[i], calib_gam.pred(y_pred[i]) = x,
        where y_true[i] ~ Bernoulli(p=x). For a perfectly-calibrated binary
        logistic model calib_gam.pred() is the identity function.

    Args:
        y_true: Binary labels in {0., 1.}
        y_pred: Predicted probabilities in [0, 1] corresponding to labels
        n_splines: Number of splines used to transform y_pred
        lam_candidates: Candidates for the penalty on the second derivative of
            the spline transformation

    Returns:
        Linearly-spaced probabilities spanning [0, 1]. Use as the binary
            logistic model's predicted probabilities on the x axis of the
            calibration plot
        Estimated actual probabilities (estimated by calib_gam) corresponding
            to the predicted probabilities above. Use as the y axis of the
            calibration plot
        Mean absolute calibration error
        The member of of lam_candidates used to fit the final calib_gam
    """
    calib_gam = GAM(
        s(0, n_splines=n_splines),
        distribution=BinomialDist(levels=1),
        link="logit"
    ).gridsearch(
        y_pred.reshape(-1, 1),
        y_true,
        lam=lam_candidates,
        progress=False
    )
    p = np.linspace(0, 1, 101)
    cal_curve = calib_gam.predict(p)
    calib_mae = mean_absolute_error(p, cal_curve)
    return p, cal_curve, calib_mae, calib_gam.terms.lam


def stratify_y_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> (np.ndarray, np.ndarray):
    """Splits predicted probabilities into those where true label is 0 and
        those where true label is 1.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities. Either has shape (n_patients,)
            representing a single prediction of y per patient, or has shape
            (n_patients, n_samples) representing n_samples predictions of y per
            patient

    Returns:
        Predicted probabilities where true label is 0
        Predicted probabilities where true label is 1
    """
    if y_pred.ndim == 1:
        return (
            y_pred[np.where(y_true == 0)[0]],
            y_pred[np.where(y_true == 1)[0]]
        )
    elif y_pred.ndim == 2:
        return (
            y_pred[:, np.where(y_true == 0)[0]],
            y_pred[:, np.where(y_true == 1)[0]]
        )
    else:
        raise NotImplementedError


def somers_dxy(y_true, y_pred):
    """Somers' Dxy simply rescales AUROC (AKA c statistic) so that Dxy = 0
        corresponds to random predictions and Dxy = 1 corresponds to
        perfect discrimination."""
    auroc = roc_auc_score(y_true, y_pred)
    return 2 * (auroc - 0.5)


def tjurs_coef(y_true, y_pred):
    """Tjur's coefficient of discrimination is the average predicted risk
        when y = 1, minus the average predicted risk when y = 0."""
    y_pred_0, y_pred_1 = stratify_y_pred(y_true, y_pred)
    return y_pred_1.mean() - y_pred_0.mean()


def score_logistic_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate several scores for logistic model performance using predicted
        risks and true labels.

        NB. We probably shouldn't report average precision - it is predicated
        on binarizing our model output but we care about risks. It is a more
        useful metric in e.g. information retrieval."""
    scores = {}
    for score_name, score_f in (
        ("C statistic", roc_auc_score),
        ("Somers' Dxy", somers_dxy),
        ("Log loss", log_loss),
        ("Brier score", brier_score_loss),
        ("Tsur's discrimination coef.", tjurs_coef),
    ):
        scores[score_name] = score_f(y_true, y_pred)
    return scores


def score_linear_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate several scores for linear model performance using predicted
        and true values."""
    scores = {}
    for score_name, score_f in (
        ("R2 score", r2_score),
        ("Mean squared error", mean_squared_error),
        ("Mean absolute error", mean_absolute_error),
        ("Median absolute error", median_absolute_error)
    ):
        scores[score_name] = score_f(y_true, y_pred)
    return scores


class Scorer:
    """Calculate confidence intervals for model evaluation scores using their
        predictions on different test folds."""

    def __init__(
        self,
        y_true: List[np.ndarray],
        y_pred: List[np.ndarray],
        scorer_function: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        n_splits: int
    ):
        """
        Args:
            y_true: Target values for each iteration
            y_pred: Predicted values for each iteration
            scorer_function: Calculates scores of the appropriate type
            n_splits: Number of train-test splits
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.scorer_function = scorer_function
        self.n_splits = n_splits
        self.scores = {
            "per_iter": {},
            "per_score": {},
            "per_score_diff": {},
            "95ci": {}
        }
        self._sanity_check()

    @property
    def n_iters(self) -> int:
        return len(self.y_true)

    @property
    def n_iters_per_split(self) -> int:
        return int(self.n_iters / self.n_splits)

    def _sanity_check(self):
        for i in range(self.n_iters):
            assert self.y_true[i].shape[0] == self.y_pred[i].shape[0]

    def calculate_scores(self):
        for i in pb(range(self.n_iters), prefix="Scorer iteration"):
            self.scores["per_iter"][i] = self.scorer_function(
                self.y_true[i], self.y_pred[i])
        self._calculate_medians_and_95ci()

    def print_scores(self, point_estimate: str, dec_places: int):
        if point_estimate == 'median':
            pe_dict = self.scores["medians"]
        elif point_estimate == 'split0':
            pe_dict = self.scores["split0"]
        else:
            raise ValueError
        for score, pe in pe_dict.items():
            print(
                f"{score} = {np.round(pe, dec_places)}",
                f"({np.round(self.scores['95ci'][score][0], dec_places)} -",
                f"{np.round(self.scores['95ci'][score][1], dec_places)})")

    def _calculate_medians_and_95ci(self):
        self.scores["medians"] = {}
        self.scores["split0"] = {}
        for score in self.scores["per_iter"][0].keys():
            self._extract_iter_per_score(score)
            self.scores["medians"][score] = np.median(
                self.scores["per_score"][score])
            self.scores["per_score_diff"][score] = (
                self.scores["per_score"][score] - self.scores["medians"][score])
            self.scores["95ci"][score] = list(
                self.scores["medians"][score] + np.percentile(
                    self.scores["per_score_diff"][score], (2.5, 97.5)))
            self.scores["split0"][score] = np.median(
                self.scores["per_score"][score][:self.n_iters_per_split])

    def _extract_iter_per_score(self, score):
        self.scores["per_score"][score] = np.zeros(self.n_iters)
        for i in range(self.n_iters):
            self.scores["per_score"][score][i] = self.scores["per_iter"][i][
                score]


class LogisticScorer(Scorer):
    """Calculate confidence intervals for logistic model evaluation scores
        using their predictions on different test folds."""

    def __init__(
        self,
        y_true: List[np.ndarray],
        y_pred: List[np.ndarray],
        scorer_function: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        n_splits: int,
        calibration_n_splines: int,
        calibration_lam_candidates: np.ndarray,
    ):
        super().__init__(y_true, y_pred, scorer_function, n_splits)
        self.calib_n_splines = calibration_n_splines
        self.calib_lam_candidates = calibration_lam_candidates
        self.p: Union[None, np.ndarray] = None
        self.calib_lams: List[List[List[float]]] = []
        self.calib_curves: List[np.ndarray] = []

    def calculate_scores(self):
        for i in pb(range(self.n_iters), prefix="Scorer iteration"):
            self.scores["per_iter"][i] = self.scorer_function(
                self.y_true[i], self.y_pred[i])
            self.p, calib_curve, calib_mae, best_lam = score_calibration(
                self.y_true[i],
                self.y_pred[i],
                self.calib_n_splines,
                self.calib_lam_candidates)
            self.scores["per_iter"][i]["Calibration MAE"] = calib_mae
            self.calib_curves.append(calib_curve)
            self.calib_lams.append(best_lam)
        self._calculate_medians_and_95ci()


class ScoreComparer:
    """Compare the per-split differences in scores of 2 models."""

    def __init__(
        self,
        scorers: Tuple[
            Union[Scorer, LogisticScorer],
            Union[Scorer, LogisticScorer]],
        scorer_names: Tuple[str, str]
    ):
        self.scorers = scorers
        self.scorer_names = scorer_names
        self.score_diff = {
            'splits': {},
            'split0': {},
            'median': {},
            '95ci': {}
        }
        self._sanity_check()

    @property
    def score_names(self) -> List[str]:
        return list(self.scorers[0].scores["per_iter"][0].keys())

    def compare_scores(self):
        for score_name in self.score_names:
            self._calculate_per_split_diff(score_name)
            self._make_separate_dict_for_split0_diff(score_name)
            self._calculate_median_diff(score_name)
            self._calculate_95ci_diff(score_name)

    def print_scores(self, point_estimate: str, dec_places: int):
        if point_estimate == 'median':
            pe_dict = self.score_diff["median"]
        elif point_estimate == 'split0':
            pe_dict = self.score_diff["split0"]
        else:
            raise ValueError
        for score, pe in pe_dict.items():
            print(
                f"{score} = {np.round(pe, dec_places)}",
                f"({np.round(self.score_diff['95ci'][score][0], dec_places)} -",
                f"{np.round(self.score_diff['95ci'][score][1], dec_places)})")

    def _sanity_check(self):
        assert isinstance(self.scorers[0], type(self.scorers[1]))
        assert len(self.scorers) == 2
        assert len(self.scorer_names) == 2
        assert set(self.score_names) == set(
            self.scorers[1].scores["per_iter"][0].keys())

    def _calculate_per_split_diff(self, score_name: str):
        self.score_diff['splits'][score_name] = (
            self.scorers[1].scores['per_score'][score_name] -
            self.scorers[0].scores['per_score'][score_name]
        )

    def _make_separate_dict_for_split0_diff(self, score_name: str):
        self.score_diff['split0'][score_name] = self.score_diff[
            'splits'][score_name][0]

    def _calculate_median_diff(self, score_name: str):
        self.score_diff['median'][score_name] = np.median(
            self.score_diff['splits'][score_name])

    def _calculate_95ci_diff(self, score_name: str):
        self.score_diff['95ci'][score_name] = np.percentile(
            self.score_diff['splits'][score_name], (2.5, 97.5))


def evaluate_samples(
    y: np.ndarray,
    y_samples: np.ndarray,
    cis: np.ndarray = np.linspace(0.95, 0.05, 20)
) -> None:
    """Generate some plots to evaluate the quality of imputed samples of
        e.g. lactate, albumin. y is (n_patients,), y_samples is
        (n_patients, n_samples per patient).

        Compares quantiles of samples to proportion of times the true value
        falls between the intervals. Note that the advantage of using our
        samples for each patient vs. sampling from the population
        distribution is that our quantiles should be narrower, so we
        measure how much narrower (on average)."""
    pop_iqrs, sam_mean_iqrs, sam_cis = [], [], []
    sam_iqrs = np.zeros((cis.shape[0], y.shape[0]))

    for i, ci in enumerate(pb(cis)):
        tail = (1 - ci) / 2
        quantiles = (tail, 1 - tail)

        pop_ci = np.quantile(y, q=quantiles)
        pop_iqrs.append(pop_ci[1] - pop_ci[0])

        sam_ci = np.quantile(y_samples, axis=1, q=quantiles).T
        sam_iqrs[i, :] = sam_ci[:, 1] - sam_ci[:, 0]
        sam_mean_iqrs.append(sam_iqrs[i, :].mean())

        sam_cis.append(
            y[np.where((y > sam_ci[:, 0]) & (y < sam_ci[:, 1]))].shape[0] /
            y.shape[0]
        )

    fig, ax = plt.subplots(2, 2, figsize=(7, 6.4))
    ax = ax.ravel()

    hist_args = {"alpha": 0.5, "density": True, "bins": 20}
    ax[0].hist(y_samples.flatten(), label="Samples", **hist_args)
    ax[0].hist(y, label="Population", **hist_args)
    ax[0].legend()

    ax[2].plot(np.linspace(0, 1), np.linspace(0, 1), alpha=0.3, c="k", ls=":")
    ax[2].scatter(cis, sam_cis, s=15)
    ax[2].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Samples CI",
        ylabel=r"Fraction of $y$ within samples CI",
    )

    ax[3].plot(cis, pop_iqrs, label="Population")
    ax[3].plot(cis, sam_mean_iqrs, label="Samples")
    ax[3].set(xlim=(0, 1), xlabel="CI", ylabel=r"Mean CI width")
    ax[3].legend()

    plt.tight_layout()
    plt.show()
