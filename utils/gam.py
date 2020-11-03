import copy
from typing import List, Union

import numpy as np
from numpy.random import RandomState
from pygam import GAM, LinearGAM, LogisticGAM
from pygam.distributions import NormalDist


def combine_mi_gams(
    gams: List[Union[LinearGAM, LogisticGAM]]
) -> Union[LinearGAM, LogisticGAM]:
    """Given a list of GAMs fitted to different datasets outputted from some
        multiple imputation procedure, combines these using Rubin's rules.

        See White et al, Section 1.2 (https://pubmed.ncbi.nlm.nih.gov/21225900/)
        for a general description of Rubin's rules, and https://bit.ly/37BgOEh
        for discussion of their application to GAMs.

        Each GAM, and the terms within it, should have the same parameters. E.g.
        spline terms for the same feature should have the same number and order
        of splines."""
    comb = copy.deepcopy(gams[0])

    # Average coefficients
    comb.coef_ = np.zeros_like(comb.coef_)
    for gam in gams:
        comb.coef_ += gam.coef_
    comb.coef_ /= len(gams)

    # Average covariance matrices
    comb.statistics_["cov"] = np.zeros_like(comb.statistics_["cov"])
    for gam in gams:
        comb.statistics_["cov"] += gam.statistics_["cov"]
    comb.statistics_["cov"] /= len(gams)

    # Apply correction factor to covariance matrix to account for
    # variation between models
    B = np.zeros_like(comb.statistics_["cov"])
    for gam in gams:
        diff = (gam.coef_ - comb.coef_).reshape(-1, 1)
        B += np.matmul(diff, diff.T)
    B /= len(gams) - 1
    comb.statistics_["cov"] += (1 + 1 / len(gams)) * B

    return comb


def quick_sample(
    gam: GAM,
    sample_at_X: np.ndarray,
    random_seed: int,
    quantity: str = "y",
    n_draws: int = 100
) -> np.ndaray:
    """
    Sample from the multivariate normal distribution over the model
    coefficients, and use the samples to predict a distribution over the target
    quantity.

    This is a simplified version of GAM.sample() as we prespecify the lam
    (regularisation penalty) on each feature instead of fitting it with
    grid search, therefore we don't consider model uncertainty resulting from
    other values of lam.

    quantity='y' is only currently supported where gam.distribution is
    Gaussian, as is the case with LinearGAM. This is because, unlike
    GAM.sample(), we don't draw samples from the model distribution using
    gam.distribution.sample() as this method is unseeded, preventing
    reproducibility. Instead, we reimplement a seeded version of sampling from
    the model distribution inside this function, but we have only done this for
    a Gaussian model distribution so far.

    Parameters
    -----------
    gam: fitted GAM object

    sample_at_X : array of shape (n_samples, m_features)
          Input data at which to draw new samples.
          Only applies for `quantity` equal to `'y'` or to `'mu`'.

    random_seed: For input to np.random.RandomState

    quantity : {'y', 'coef', 'mu'}, default: 'y'
        What quantity to return pseudorandom samples of.

    n_draws : positive int, optional (default=100)
        The number of samples to draw from distribution over the model
            coefficients

    Returns
    -------
    draws : 2D array of length n_draws
        Simulations of the given `quantity` using samples from the
        posterior distribution of the coefficients given the response data.
        Each row is a pseudorandom sample.

        If `quantity == 'coef'`, then the number of columns of `draws` is
        the number of coefficients (`len(self.coef_)`).

        Otherwise, the number of columns of `draws` is the number of
        rows of `X`.
    """
    if quantity not in {"mu", "coef", "y"}:
        raise ValueError(
            "`quantity` must be one of 'mu', 'coef', 'y';" f" got {quantity}"
        )

    rnd = RandomState(random_seed)
    coef_draws = rnd.multivariate_normal(
        gam.coef_, gam.statistics_["cov"], size=n_draws
    )

    if quantity == "coef":
        return coef_draws

    linear_predictor = gam._modelmat(sample_at_X).dot(coef_draws.T)
    mu_shape_n_draws_by_n_samples = gam.link.mu(linear_predictor,
                                                gam.distribution).T
    if quantity == "mu":
        return mu_shape_n_draws_by_n_samples
    else:
        if isinstance(gam.distribution, NormalDist):
            scale = gam.distribution.scale
            standard_deviation = scale ** 0.5 if scale else 1.0
            return rnd.normal(
                loc=mu_shape_n_draws_by_n_samples,
                scale=standard_deviation,
                size=None
            )
        else:
            raise NotImplementedError
