import numpy as np
from scipy import stats


class TruncatedDistribution:
    """Thin wrapper around a frozen scipy continuous variable to allow
        truncation."""
    def __init__(
        self,
        rv: stats._distn_infrastructure.rv_frozen,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        random_seed=None
    ):
        self.rv = rv
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rnd = np.random.RandomState(random_seed)

    @property
    def normaliser(self) -> float:
        return self.rv.cdf(self.upper_bound) - self.rv.cdf(self.lower_bound)

    def sample(self, n_samples: int) -> np.ndarray:
        pass
