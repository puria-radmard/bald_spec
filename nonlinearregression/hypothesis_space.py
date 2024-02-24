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
        import pdb; pdb.set_trace(header = 'proof some shapes here')
        return self.prior.log_prob(thetas)

    def __contains__(self, theta: _T):
        return self.prior.support.check(theta).all()
