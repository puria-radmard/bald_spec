import torch
from torch import Tensor as _T
import torch.distributions as d
from torch.distributions import Distribution as _D

from typing import Type
from abc import ABC, abstractmethod


class NoiseModelBase(ABC):
    """
    This class needs to both evaluate model likelihood given an f evaluation,
    and be able to sample from an output given the f sample

    As such, it needs to accept a TYPE of distribution, which it will be able
        to instantiate using only the output of the f function...
    """

    distribution_type: Type[_D]
    data_dimensionality: int

    def __init__(self, distribution_type: Type[_D], data_dimensionality: int) -> None:
        self.distribution_type = distribution_type
        self.data_dimensionality = data_dimensionality
        super().__init__()

    @abstractmethod
    def instantiate_distribution(self, regression_evaluation: _T) -> _D:
        raise NotImplementedError

    def log_likelihood(self, labels: _T, regression_evaluation: _T) -> _T:
        import pdb; pdb.set_trace()
        assert labels.shape == regression_evaluation.shape, "labels y and evaluations f(x) should exist in the same space!"
        dist = self.instantiate_distribution(regression_evaluation)
        return dist.log_prob(labels)
    
    def sample_output(self, sample_num: int, regression_evaluation: _T) -> _T:
        "Tacks dimension to end of tensor"
        assert regression_evaluation.shape[-1] == self.data_dimensionality
        dist = self.instantiate_distribution(regression_evaluation)
        return dist.sample([sample_num]).movedim(0, -1)


class UnitRateGammaNoiseModel(NoiseModelBase):
    """
    Gamma with alpha = f, beta = 1
    """
    def __init__(self, data_dimensionality = 1) -> None:
        super().__init__(d.Gamma, data_dimensionality)

    def instantiate_distribution(self, regression_evaluation: _T) -> _D:
        return self.distribution_type(
            concentration = regression_evaluation, 
            rate = torch.ones_like(regression_evaluation).float()
        )
