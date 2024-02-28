"""
Monte Carlo estimation of the parameters + a emission distribution for each one
    means we have an ensemble predictive distribution over each datapoint.
Here, we visualise that distribution for a given test location.

Further, we have to take Monte Carlo samples from each of these distributions to 
    calculate the ensemble entropy, e.g. for BALD
We will scatter these samples onto the regression line.
    Colouring the density of these samples will surrogate a sort of `confidence interval'
        for the hypothesis posterior
"""

import torch
from typing import Union
from torch import Tensor as _T

import numpy as np

from noisemodels.base import NoiseModelBase as _NM
from nonlinearregression.mean_fit import NonLinearRegressionModelBase as _NR

from matplotlib.pyplot import Axes


def ensemble_predictive_distribution_at_testpoint(
    ax: Union[Axes, None],
    function_eval_at_test_point: _T,
    noise_model: _NM,
    sampled_noisy_testpoint_labels_at_test_point: _T,
    y_count = 1000, y_min=None, y_max=None, hist_num_bins = 50
) -> np.ndarray:
    """
    To visualise the actual ensemble predictive distribution:
        function_eval_at_test_point of shape [I, dim(out)], 
        and is a slice of the "function_eval" variable in BALDMCAcquisitionFunction.evaluate_function_on_batch

        We construct a grid of y_points and pass the two to noise_model.log_likelihood

    To visualise the fidelity of the MC samples from the ensemble predictive distribution:
        sampled_noisy_testpoint_labels_at_test_point of shape [K_i, I, dim out],
            and is a slice of the "sampled_noisy_testpoint_labels" variable in BALDMCAcquisitionFunction.evaluate_function_on_batch
            - first axis indexes emissions probabilities of different ensemble members (MC samples)
            - second axis indexes different ensemble members (MCMC samples) of the parameters from the hypothesis
            - third axis typically = 1 (enforced here)
    
    y_count gives the number of output values for which the ensemble predictive
        distribution is evaluated (a y-slice at the x-testpoint)
    
    if y_max is not given, use the maximum value found in 
        sampled_noisy_testpoint_labels_at_test_point
    """

    assert noise_model.data_dimensionality == 1, "Cannot plot ensemble_predictive_distribution_at_testpoint for multidimensional labels yet"
    if y_max == None:
        y_max = sampled_noisy_testpoint_labels_at_test_point.max()
    if y_min == None:
        y_min = sampled_noisy_testpoint_labels_at_test_point.min()
    y_tests = torch.linspace(y_min, y_max, y_count).unsqueeze(-1)   # now shaped [B, 1] where B i y_count

    function_eval_at_test_point = function_eval_at_test_point.unsqueeze(0).repeat(y_count, 1, 1)    # Same x testpoint so same evaluation!
    likelihood = noise_model.log_likelihood(y_tests, function_eval_at_test_point).exp()
    predictive_distribution = likelihood.mean(-1).numpy()

    if ax is not None:
        ax.plot(y_tests.flatten().numpy(), predictive_distribution, label = 'True MC approximation ensemble predictive distribution')
        ax.hist(
            sampled_noisy_testpoint_labels_at_test_point.flatten().numpy(),
            hist_num_bins, label = 'Histogram of MC samples from ensemble emission distributions',
            density=True, alpha = 0.3
        )
        
        y_min, y_max = ax.get_ylim()
        ax.scatter(
            sampled_noisy_testpoint_labels_at_test_point.flatten().numpy(),
            y_min * np.ones_like(sampled_noisy_testpoint_labels_at_test_point.flatten().numpy()) + 0.25 * (y_max - y_min),
            marker = 'x', s = 1, color = 'red',
            label = 'MC samples from ensemble emission distributions'
        )
        ax.set_ylim([y_min, y_max])

    return y_tests.flatten().numpy(), predictive_distribution



def regression_mean_line_with_emission_distributions(
    ax: Axes, all_sampled_params: _T, reg_model: _NR, noise_model: _NM,
    x_data, y_data, x_min: int = None, x_max: int = None, x_count = 1000, 
):
    if x_max == None:
        x_max = x_data.max()
    if x_min == None:
        x_min = x_data.min()

    # See labels for explanations...
    x_axis = torch.linspace(x_min, x_max, x_count).unsqueeze(1)
    mean_params = all_sampled_params.mean(0, keepdim=True)
    function_eval_at_mean_params = reg_model.evaluate_f(x_axis, mean_params).squeeze()
    ax.plot(x_axis.numpy(), function_eval_at_mean_params.numpy(), label = '$f(E[\\theta])$')

    gridded_all_function_eval = reg_model.evaluate_f(x_axis, all_sampled_params)
    mean_function_eval = gridded_all_function_eval.mean(1).squeeze()
    ax.plot(x_axis.numpy(), mean_function_eval.numpy(), label = '$E[f(\\theta)]$')

    y_max=function_eval_at_mean_params.max() * 1.5
    for i, x in enumerate(x_axis):
        y_axis, predictive_distribution_at_x = ensemble_predictive_distribution_at_testpoint(
            None, gridded_all_function_eval[i], noise_model,
            None, y_min=0.0, y_max=y_max, y_count=100
        )
        x_rep = x.repeat(len(predictive_distribution_at_x)).numpy()
        renormed_predictive_distribution_at_x = predictive_distribution_at_x / predictive_distribution_at_x.max()
        ax.scatter(x_rep, y_axis, c = renormed_predictive_distribution_at_x, alpha = 0.1)
    
    ax.scatter(x_data, y_data, marker = 'x', label = 'Collected data')

