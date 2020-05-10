# -*- coding: utf-8 -*-
from typing import Tuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from progressbar import progressbar
from pygam import GAM
from pygam.distributions import BinomialDist
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                             mean_absolute_error)

from .constants import (N_GAM_CONFIDENCE_INTERVALS,
                        GAM_OUTER_CONFIDENCE_INTERVALS)


def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bs_iters: int = 1000,
        dec_places: int = 8,
        n_cis: int = N_GAM_CONFIDENCE_INTERVALS,
        outer_cis: Tuple[float] = GAM_OUTER_CONFIDENCE_INTERVALS
):
    """Evaluates model predictions by calculating scores with bootstrap
        confidence intervals. Plots and scores model calibration error."""
    bss = BootstrapScorer(y_true, y_pred, bs_iters)
    bss.calculate_scores()
    p, _, calib_cis, calib_mae = score_calibration(
        y_true, y_pred, n_cis, outer_cis)
    plot_calibration(p, calib_cis)
    bss.print_scores(dec_places)
    print(f'Calibration MAE = {np.round(calib_mae, dec_places)}')


def score_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                      n_cis: int, outer_cis: Tuple[float],
                      lam_candidates: np.ndarray = np.logspace(1.5, 2.5)):
    """Derive smooth model calibration curve using a GAM. Report calibration
        error versus line of indentity."""
    calib_gam = GAM(distribution=BinomialDist(levels=1),
                    link='logit').gridsearch(y_pred.reshape(-1, 1),
                                             y_true,
                                             lam=lam_candidates)
    p = np.linspace(0, 1, 101)
    cal_curve = calib_gam.predict(p)
    calib_mae = mean_absolute_error(p, cal_curve)
    calib_cis = calib_gam.confidence_intervals(
        p, quantiles=np.linspace(*outer_cis, n_cis * 2))
    return p, cal_curve, calib_cis, calib_mae


def plot_calibration(p: np.ndarray, calib_cis: np.ndarray):
    """Plot calibration curve, with confidence intervals."""
    f, ax = plt.subplots(figsize=(4, 4))
    n_cis = int(calib_cis.shape[1] / 2)
    for k in range(n_cis):
        ax.fill_between(p, calib_cis[:, k], calib_cis[:, -(k + 1)],
                        alpha=1 / n_cis, color='black', lw=0.0)
    ax.plot([0, 1], [0, 1], linestyle='dotted', c='red')
    ax.set(xlabel='Predicted risk', ylabel='True risk',
           xlim=[0, 1], ylim=[0, 1], title='Calibration')
    plt.show()


def somers_dxy(y_true, y_pred):
    """Somers' Dxy simply rescales AUROC / c statistic so that Dxy = 0
        corresponds to random predictions and Dxy = 1 corresponds to
        perfect discrimination."""
    auroc = roc_auc_score(y_true, y_pred)
    return 2 * (auroc - 0.5)


def tjurs_cof(y_true, y_pred):
    """Tjur's coefficient of discrimination is the average predicted risk
        when y = 1, minus the average predicted risk when y = 0."""
    strata = []
    for y in (0, 1):
        strata.append(y_pred[np.where(y_true == y)].mean())
    return strata[1] - strata[0]


def score_predictions(y_true, y_pred):
    """Calculate several scores for model performance using predicted risks
        and true labels.

        NB. We probably shouldn't report average precision - it is predicated
        on binarizing our model output but we care about risks. It is a more
        useful metric in e.g. information retrieval."""
    scores = {}
    for score_name, score_f in (
            ('C statistic', roc_auc_score),
            ("Somers' Dxy", somers_dxy),
            ('Log loss', log_loss),
            ('Brier score', brier_score_loss),
            ("Tsur's discrimination coef.", tjurs_cof)
    ):
        scores[score_name] = score_f(y_true, y_pred)
    return scores


class BootstrapScorer:
    """Calculate confidence intervals for model evaluation scores by drawing
        bootstrap samples from true labels and predicted risks.

        TODO: Consider retiring this class, given that the CIs it calculates
            don't capture some important sources of uncertainty in our analysis,
            e.g. that which stems from our splitting of the dataset.
    """

    def __init__(self, y_true, y_pred, bootstrap_iter, random_seed=1):
        self.y_true = y_true
        self.y_pred = y_pred
        self.bootstrap_iter = bootstrap_iter
        self.rnd = np.random.RandomState(random_seed)
        self.scores = {'bs_iter': {}, 'bs': {}, '95ci': {}}
        self.y_index = np.arange(y_true.shape[0])
        warn("""The bootstrap confidence intervals derived here ignore
             important sources of uncertainty, e.g. that which stems from
             data splitting. Fix this ASAP!""")

    def calculate_scores(self):
        self.scores['point_estimates'] = score_predictions(
            self.y_true, self.y_pred)
        for i in progressbar(range(self.bootstrap_iter),
                             prefix='Bootstrapping:'):
            self._run_bootstrap_iter(i)
        self._calculate_95ci()

    def print_scores(self, dec_places):
        for score, pe in self.scores['point_estimates'].items():
            print(f"{score} = {np.round(pe, dec_places)}",
                  f"({np.round(self.scores['95ci'][score][0], dec_places)} -",
                  f"{np.round(self.scores['95ci'][score][1], dec_places)})")

    def _run_bootstrap_iter(self, i):
        self.scores['bs_iter'][i] = score_predictions(
            *self._make_bootstrap_fold())

    def _make_bootstrap_fold(self):
        bs_index = self.rnd.choice(self.y_index,
                                   size=self.y_index.shape[0])
        return (np.array([self.y_true[i] for i in bs_index]),
                np.array([self.y_pred[i] for i in bs_index]))

    def _calculate_95ci(self):
        for score in self.scores['point_estimates'].keys():
            self._extract_iter_per_score(score)
            self.scores['bs'][score] = (
                    self.scores['bs'][score] -
                    self.scores['point_estimates'][score])
            self.scores['95ci'][score] = list(
                self.scores['point_estimates'][score] +
                np.percentile(self.scores['bs'][score], (2.5, 97.5)))

    def _extract_iter_per_score(self, score):
        self.scores['bs'][score] = np.zeros(self.bootstrap_iter)
        for i in range(self.bootstrap_iter):
            self.scores['bs'][score][i] = self.scores['bs_iter'][i][score]


def evaluate_samples(y: np.ndarray, y_samples: np.ndarray,
                     cis: np.ndarray = np.linspace(0.95, 0.05, 20)) -> None:
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

    for i, ci in enumerate(progressbar(cis)):
        tail = (1 - ci) / 2
        quantiles = (tail, 1 - tail)

        pop_ci = np.quantile(y, q=quantiles)
        pop_iqrs.append(pop_ci[1] - pop_ci[0])

        sam_ci = np.quantile(y_samples, axis=0, q=quantiles).T
        sam_iqrs[i, :] = sam_ci[:, 1] - sam_ci[:, 0]
        sam_mean_iqrs.append(sam_iqrs[i, :].mean())

        sam_cis.append(y[np.where((y > sam_ci[:, 0]) &
                                  (y < sam_ci[:, 1]))].shape[0] / y.shape[0])

    fig, ax = plt.subplots(2, 2, figsize=(7, 6.4))
    ax = ax.ravel()

    hist_args = {'alpha': 0.5, 'density': True, 'bins': 20}
    ax[0].hist(y_samples.flatten(), label='Samples', **hist_args)
    ax[0].hist(y, label='Population', **hist_args)
    ax[0].legend()

    ax[2].plot(np.linspace(0, 1), np.linspace(0, 1),
               alpha=0.3, c='k', ls=':')
    ax[2].scatter(cis, sam_cis, s=15)
    ax[2].set(xlim=(0, 1), ylim=(0, 1),
              xlabel='Samples CI',
              ylabel=r'Fraction of $y$ within samples CI')

    ax[3].plot(cis, pop_iqrs, label='Population')
    ax[3].plot(cis, sam_mean_iqrs, label='Samples')
    ax[3].set(xlim=(0, 1), xlabel='CI', ylabel=r'Mean CI width')
    ax[3].legend()

    plt.tight_layout()
    plt.show()
