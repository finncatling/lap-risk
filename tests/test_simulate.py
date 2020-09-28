import pytest
from scipy.stats import norm, exponpow
from utils.simulate import TruncatedDistribution


class TestTruncatedDistribution:
    # @pytest.fixture(scope='class', params=[
    #     (norm(loc=0, scale=1), -10, 0.3),
    #     (exponpow(b=75, loc=71, scale=50), 18, 110)])
    # def truncated_distribution(self, rv, lower_bound, upper_bound):
    #     return TruncatedDistribution(rv, lower_bound, upper_bound)

    @pytest.fixture(scope='class')
    def truncated_distribution(self):
        return TruncatedDistribution(exponpow(b=75, loc=71, scale=50), 18, 110)

    def test_normaliser(self, truncated_distribution):
        assert truncated_distribution.normaliser <= 1
        assert truncated_distribution.normaliser >= 0
