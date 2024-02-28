from typing import Union, Dict
from abc import ABC, abstractmethod

from torch import Tensor as _T

from data.base import ActiveLearningDataLoaderBase as _DL
from hypothesis_space.hypothesis_space import HypothesisSpaceBase as _H
from nonlinearregression.mean_fit import NonLinearRegressionModelBase as _NL
from noisemodels.base import NoiseModelBase as _NM


class ParameterMCMCBase(ABC):
    """
    Just the basic interactions - each class will build on this a LOT
    """

    def __init__(
        self,
        hypothesis_space: _H,
        data_loader: _DL,
        reg_model: _NL,
        noise_model: _NM,
    ) -> None:
        self.hypothesis_space = hypothesis_space
        self.data_loader = data_loader
        self.reg_model = reg_model
        self.noise_model = noise_model

    def initialise_chains(
        self,
        initial_hypotheses: Union[_T, None],
        num_chains: Union[int, None],
    ):
        assert (initial_hypotheses is None) != (num_chains is None), "Must specify exactly one of initial_hypotheses or I"
        if num_chains is None:
            assert initial_hypotheses in self.hypothesis_space
            num_chains = initial_hypotheses.shape[0]
        else:
            initial_hypotheses = self.hypothesis_space.sample_from_prior(num_chains)
        return initial_hypotheses, num_chains

    @abstractmethod
    def step_sample(
        self, 
        initial_hypotheses: Union[_T, None],    # if not provided, an initial theta will be sampled from the hypothesis space <num_chains> times
        num_chains: Union[int, None],
        *args, 
        **kwargs
    ) -> Dict[str, _T]:
        """
        Should return a dictionary, with one of the keys being 'next_hypotheses'
        It is important that this one is shaped [num_chains, dim(H)]
        """
        raise NotImplementedError

    @abstractmethod
    def sample_many(
        self, 
        num_steps: int,
        initial_hypotheses: Union[_T, None],    # if not provided, an initial theta will be sampled from the hypothesis space <num_chains> times
        num_chains: Union[int, None],
        *args, 
        **kwargs
    ) -> Dict[str, _T]:
        """
        Should return a dictionary, with one of the keys being 'all_samples'
        It is important that this one is shaped [num_steps, num_chains, dim(H)]
        """
        raise NotImplementedError

    def evaluate_unnormalised_log_posterior(
        self,
        thetas: _T,                 # [num theta (could be I or num_chains!), dim(H)]
    ):
        result = self.hypothesis_space.log_prior(thetas)                         # [num theta]

        self.data_loader.set_data_mode('train')
        for training_batch in self.data_loader:
            
            data = training_batch['data']                                   # [B, dim_in]
            labels = training_batch['labels']                               # [B, dim_out]

            model_eval = self.reg_model.evaluate_f(data, thetas)                 # [B, num theta, dim_out]
            likelihoods = self.noise_model.log_likelihood(labels, model_eval)    # [B, num theta]

            result += likelihoods.sum(0)                                  # [num theta]

        return result

