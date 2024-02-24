from torch import Tensor as _T
from abc import ABC, abstractmethod

from torch import Tensor as _T
from data.base import ActiveLearningDataLoaderBase as _DL
from nonlinearregression.hypothesis_space import HypothesisSpaceBase as _H
from nonlinearregression.mean_fit import NonLinearRegressionModelBase as _NL
from noisemodels.base import NoiseModelBase as _NM


class ParameterMCMCBase:
    """
    Just the basic interactions - each class will build on this a LOT
    """

    def __init__(self, hypothesis_space: _H) -> None:
        self.hypothesis_space = hypothesis_space

    @abstractmethod
    def step_sample(self, *args, **kwargs) -> _T:
        raise NotImplementedError

    @abstractmethod
    def step_many(self, *args, **kwargs) -> _T:
        raise NotImplementedError

    @classmethod
    def evaluate_unnormalised_log_posterior(
        thetas: _T,                 # [num theta, dim(H)]
        dataset: _DL,
        hypothesis_space: _H,
        reg_model: _NL,
        noise_model: _NM,
    ):
        result = hypothesis_space.log_prior(thetas)  # [num theta]

        dataset.train()
        for training_batch in dataset:
            
            # XXX: shapes!
            data = training_batch['data']                   # [B, dim_in]
            labels = training_batch['labels']               # [B, dim_out]
        
            model_eval = reg_model.evaluate_f(data, thetas) # [num theta, B, 1]
            likelihoods = noise_model.log_likelihood(labels, model_eval)    # XXX shape

            result += likelihoods.sum(555)

        return result

