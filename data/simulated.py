import torch
from torch import Tensor as _T

from random import shuffle

from data.base import ActiveLearningDataLoaderBase
from hypothesis_space.hypothesis_space import HypothesisSpaceBase as _H
from nonlinearregression.mean_fit import NonLinearRegressionModelBase as _NL
from noisemodels.base import NoiseModelBase as _NM

class SimulatedDataLoader(ActiveLearningDataLoaderBase):

    """
    For each input_data:
        sample a theta from gen_hypothesis_space's prior,
        evaluate f with reg_model
        add noise to it with noise_model
    
    Obviously, if fitting a model to this, you should have gen_hypothesis_space's ("prior") distribution 
        be different to the prior you start with in the fitting hypothesis space

        parameter_sampler is model agnostic

        reg_model should be the same if you're directly comparing the fitted posterior with gen_hypothesis_space's prior

        noise_model can be the same or different depending on experimental condition...
    """

    def __init__(
        self, 
        input_data: _T,                 # of shape [N, output dimensionality (1)]
        test_input_data: _T,                 # of shape [N_test, output dimensionality (1)]
        gen_hypothesis_space: _H,      # Along with parameter_sampler, needed to sample some theta This should be diffe
        reg_model: _NL,                 # Needed to evaluate f at x = input_data with theta
        noise_model: _NM,               # Needed to add noise to f
        num_hypothesis: int,
        training_batch_size: int, 
        querying_batch_size: int, 
        initial_labelled_dataset_size: int
        ) -> None:

        # Generate all data
        assert input_data.shape[0] >= initial_labelled_dataset_size
        assert len(input_data.shape) == 2
        labels, test_labels = self.generate_data(
            input_data, test_input_data,
            num_hypothesis, gen_hypothesis_space, reg_model, noise_model
        )
        
        # Partition dataset
        all_indices = list(range(input_data.shape[0]))
        shuffle(all_indices)
        training_indices = set(all_indices[:initial_labelled_dataset_size])

        super().__init__(
            input_data, labels, test_input_data, test_labels, 
            training_indices, training_batch_size, querying_batch_size
        )


    def generate_data(
        self, input_data: _T, test_input_data: _T, num_hypothesis: int, gen_hypothesis_space: _H, reg_model: _NL, noise_model: _NM
    ) -> _T:

        all_data = torch.concat([input_data, test_input_data], dim = 0)

        # Sample a few hypotheses and evaluate at data
        thetas = gen_hypothesis_space.sample_from_prior(num_hypothesis)     # [I, dim(H)]
        mean_output = reg_model.evaluate_f(all_data, thetas)  # [B, I, data dim (1)]

        # Add noise to the model
        actual_output = noise_model.sample_output(1, mean_output)      # [B, num noise samples (1), num_hypothesis, data dim (1)]

        # Randomly select from hypotheses - this is bespoke to this class
        selected_hypotheses = torch.randint(0, num_hypothesis, [all_data.shape[0]])
        selected_outputs = actual_output.squeeze(1)[range(len(selected_hypotheses)), selected_hypotheses]
        
        # self.selected_hypotheses = selected_hypotheses
        # self.thetas = thetas

        selected_train_outputs = selected_outputs[:input_data.shape[0]]
        selected_test_outputs = selected_outputs[input_data.shape[0]:]

        return selected_train_outputs, selected_test_outputs

