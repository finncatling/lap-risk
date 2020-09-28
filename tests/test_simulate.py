import pytest
from scipy.stats import norm, exponpow
from utils.simulate import TruncatedDistribution


class TestTruncatedDistribution:
    @pytest.fixture(scope='class', params=[
        (norm(loc=0, scale=1), -10, 0.3),
        (exponpow(b=75, loc=71, scale=50), 18, 110)])
    def truncated_distribution(self, request):
        return TruncatedDistribution(*request.param)

    def test_normaliser(self, truncated_distribution):
        assert truncated_distribution.normaliser <= 1
        assert truncated_distribution.normaliser >= 0

    def test_sample(self, truncated_distribution):
        n_samples = 10000
        samples = truncated_distribution.sample(n_samples)
        assert samples.shape[0] == n_samples
        assert samples.min() >= truncated_distribution.lower_bound
        assert samples.max() <= truncated_distribution.upper_bound
