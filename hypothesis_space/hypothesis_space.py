from torch import Tensor as _T
from torch.distributions import Distribution as _D

class HypothesisSpaceBase:
    """
    $\mathcal{H}$
    """

    def __init__(self, prior: _D, dimensionality: int) -> None:
        self.prior = prior
        self.dimensionality = dimensionality

    def sample_from_prior(self, sample_num: int) -> _T:
        samples = self.prior.sample([sample_num])
        assert list(samples.shape) == [sample_num, self.dimensionality], f"prior did not return {self.dimensionality}-shaped thetas"
        return samples

    def log_prior(self, thetas: _T) -> _T:
        assert list(thetas.shape) == [thetas.shape[0], self.dimensionality]
        return self.prior.log_prob(thetas).sum(-1)  # [len(thetas)]

    def __contains__(self, theta: _T):
        assert len(theta.shape) == 2
        return self.prior.support.check(theta).all()
