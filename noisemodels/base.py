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
        """
        labels of shape [batch, output dimensionality]
        regression_evaluation of shape [batch, I, output dimensionality]

        returns of shape [batch, I], as one would expect
        """
        assert self.data_dimensionality == labels.shape[-1] == regression_evaluation.shape[-1], "labels y and evaluations f(x) should exist in the same space!"
        assert len(labels.shape) == 2 and len(regression_evaluation.shape) == 3
        dist = self.instantiate_distribution(regression_evaluation)
        return dist.log_prob(labels.unsqueeze(1)).sum(-1)

    def log_likelihood_many(self, sampled_ensemble_labels: _T, regression_evaluation: _T) -> _T:
        """
        Bespoke version of self.log_likelihood for the sake of the ensemble MC BALD estimator

        sampled_ensemble_labels now of shape [K_i (number of samples from each ensemble member), batch, I, output dimensionality]
        """
        assert self.data_dimensionality == sampled_ensemble_labels.shape[-1] == regression_evaluation.shape[-1], "sampled_ensemble_labels y and evaluations f(x) should exist in the same space!"
        assert len(sampled_ensemble_labels.shape) == 4 and len(regression_evaluation.shape) == 3
        B = sampled_ensemble_labels.shape[1]
        dist = self.instantiate_distribution(regression_evaluation)
        return dist.log_prob(sampled_ensemble_labels).sum(-1)
    
    def sample_output(self, sample_num: int, regression_evaluation: _T) -> _T:
        "Tacks dimension to end of tensor"
        assert regression_evaluation.shape[-1] == self.data_dimensionality
        dist = self.instantiate_distribution(regression_evaluation)
        return dist.sample([sample_num]) # dim appended to start of shape!

    def ensemble_entropies(self, regression_evaluation: _T) -> _T:
        assert regression_evaluation.shape[-1] == self.data_dimensionality
        dist = self.instantiate_distribution(regression_evaluation)
        return dist.entropy()


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
