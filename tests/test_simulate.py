import pytest
from scipy.stats import norm, exponpow
from utils.simulate import TruncatedDistribution


@pytest.mark.parametrize(
    'rv, lower_bound, upper_bound',
    [(norm(loc=0, scale=1), -10, 0.3),
     (exponpow(b=75, loc=71, scale=50), 18, 110)]
)
class TestTruncatedDistribution:
    def test_sample(self, rv, lower_bound, upper_bound):
        td = TruncatedDistribution(rv, lower_bound, upper_bound)
        assert td.normaliser <= 1
        assert td.normaliser >= 0
