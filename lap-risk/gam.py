# -*- coding: utf-8 -*-
import copy
import numpy as np
from pygam import GAM
from typing import List
from numpy.random import RandomState


def combine_mi_gams(gams: List[GAM]) -> GAM:
    """Given a list of GAMs fitted to different datasets
        outputted from some multiple imputation procedure,
        combines these using Rubin's rules. See
        https://bit.ly/37BgOEh for more details.

        Each GAM, and the terms within it, should have
        the same parameters. E.g. spline terms for the
        same feature should have the same number and
        order of splines."""
    comb = copy.deepcopy(gams[0])
    
    # Average coefficients
    comb.coef_ = np.zeros_like(comb.coef_)
    for gam in gams:
        comb.coef_ += gam.coef_
    comb.coef_ /= len(gams)

    # Average covariance matrices
    comb.statistics_['cov'] = np.zeros_like(comb.statistics_['cov'])
    for gam in gams:
        comb.statistics_['cov'] += gam.statistics_['cov']
    comb.statistics_['cov'] /= len(gams)

    # Apply correction factor to covariance matrix to account for
    # variation between models
    B = np.zeros_like(comb.statistics_['cov'])
    for gam in gams:
        diff = (gam.coef_ - comb.coef_).reshape(-1, 1)
        B += np.matmul(diff, diff.T)
    B /= len(gams) - 1
    comb.statistics_['cov'] += (1 + 1 / len(gams)) * B
    
    return comb


def quick_sample(gam, sample_at_X, random_seed, quantity='y', n_draws=100):
        """Sample from the multivariate normal distribution
            over the model coefficients, and use the samples to predict a
            distribution over the target quantity.

        TODO: Investigate why the default .sample() method in pygam is more
            involved than this

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
            posterior distribution of the coefficients and smoothing parameter
            given the response data. Each row is a pseudorandom sample.

            If `quantity == 'coef'`, then the number of columns of `draws` is
            the number of coefficients (`len(self.coef_)`).

            Otherwise, the number of columns of `draws` is the number of
            rows of `X`.
        """
        if quantity not in {'mu', 'coef', 'y'}:
            raise ValueError("`quantity` must be one of 'mu', 'coef', 'y';"
                             " got {}".format(quantity))

        rnd = RandomState(random_seed)
        coef_draws = samples = rnd.multivariate_normal(gam.coef_,
                                                       gam.statistics_['cov'],
                                                       size=n_draws)

        if quantity == 'coef':
            return coef_draws

        linear_predictor = gam._modelmat(sample_at_X).dot(coef_draws.T)
        mu_shape_n_draws_by_n_samples = gam.link.mu(
            linear_predictor, gam.distribution).T
        if quantity == 'mu':
            return mu_shape_n_draws_by_n_samples
        else:
            return gam.distribution.sample(mu_shape_n_draws_by_n_samples)
