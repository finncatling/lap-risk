import numpy as np
from scipy import stats


class TruncatedDistribution:
    """Thin wrapper around a frozen scipy continuous random variable to allow
        truncation of samples from that variable.

        Adapted from https://stackoverflow.com/a/11492527/1684046 """

    def __init__(
        self,
        rv: stats._distn_infrastructure.rv_frozen,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        random_state: np.random.RandomState = None
    ):
        """
        Args:
            rv: Frozen scipy continuous random variables, e.g. norm(0, 1)
            lower_bound: Samples should be >= this value
            upper_bound: Samples should be <= this value
            random_state: Pass this for deterministic sampling
        """
        self.rv = rv
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if random_state is None:
            self.rnd = np.random.RandomState()
        else:
            self.rnd = random_state

    @property
    def lower_quantile(self) -> float:
        return self.rv.cdf(self.lower_bound)

    @property
    def normaliser(self) -> float:
        return self.rv.cdf(self.upper_bound) - self.lower_quantile

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Args:
            n_samples: Number of samples from truncated random variables

        Returns:
            1D ndarray of samples
        """
        quantiles = (self.rnd.random_sample(n_samples) *
                     self.normaliser +
                     self.lower_quantile)
        return self.rv.ppf(quantiles)
