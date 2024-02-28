import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import Tensor as _T

from utils.logging_utils import configure_save_path
from utils.visualisation_utils import (
    ensemble_predictive_distribution_at_testpoint, 
    regression_mean_line_with_emission_distributions
)
from utils.performance_utils import evaluate_test_marginal_log_likelihood


from run_jexciton_experiments.initialise_acquisiton import *
from run_jexciton_experiments.initialise_inference import *     # data_loader implicit here!

from run_jexciton_experiments.initialise_dataset import dataset_type, gen_hypothesis_space

# Some set up stuff...
save_path = configure_save_path(base = args.save_path, max_dirs = 5000)
with open(os.path.join(save_path, 'args.json'), 'w') as f:
    json.dump(vars(args), f)

initial_parameter_samples = None
initial_parameter_unnormalised_log_posterior = None

all_test_marginal_log_likelihoods = []

x_min, x_max = float('inf'), float('-inf')
y_min, y_max = float('inf'), float('-inf')


# Prior data
for t in tqdm(range(num_acquisition_rounds)):

    print(f'Acquisition round {t}')

    # Initialise or 'resume' sampling... this might not be optimal though!
    print(f'\tPerforming MCMC sampling in parameter space')
    theta_sample_dict = mcmc_obj.sample_many(
        num_steps = num_steps, 
        num_chains = num_chains if initial_parameter_samples is None else None,
        initial_hypotheses=initial_parameter_samples, 
    )

    parameter_samples = theta_sample_dict['all_samples']                                          # [num_steps, num_chains, dim(H)]
    parameter_unnormalised_log_posterior = theta_sample_dict['all_unnormalised_log_posterior']    # [num_steps, num_chains]
    np_parameter_unnormalised_log_posterior = parameter_unnormalised_log_posterior.numpy()

    # Set up for next time
    initial_parameter_samples = parameter_samples[-1]

    # Get the acquisition score for each datapoint in the unlabelled set
    print(f'\tEvaluating acquision function on query set')
    flattened_parameters = parameter_samples.reshape(-1, mcmc_obj.hypothesis_space.dimensionality)
    acq_dict = acquisition_function.evaluate_function_on_data_loader(
        data_loader = data_loader,
        all_parameter_estimates = flattened_parameters,
        use_tqdm=True
    )
    querying_set_scores = acq_dict['scores']
    querying_set_indices = acq_dict['indices']


    # Get the top <data_points_per_acquision_round> indices
    top_scoring_indices = querying_set_scores.argsort(descending = True)[:num_data_points_per_acquision_round]
    chosen_indices = querying_set_indices[top_scoring_indices]

    # Evaluate on the test set
    print(f'\tEvaluating marginal log-likelihood on test set')
    new_test_loglikelihood = evaluate_test_marginal_log_likelihood(
        all_sampled_params = flattened_parameters,
        reg_model = reg_model,
        noise_model = noise_model,
        data_loader = data_loader
    )
    all_test_marginal_log_likelihoods.append(new_test_loglikelihood.item())


    ############################################ Visualisations for this round
    print(f'\tVisualising results of round')
    plt.cla()

    # Plot the emissions_distribution at one testpoint
    fig, axes = plt.subplots(1)
    test_point_index = chosen_indices[0]
    converted_index = acq_dict['indices'].tolist().index(test_point_index)
    selected_function_eval = acq_dict['function_eval'][converted_index]
    selected_sampled_noisy_testpoint_labels = acq_dict['sampled_noisy_testpoint_labels'][converted_index]
    ensemble_predictive_distribution_at_testpoint(axes, selected_function_eval, noise_model, selected_sampled_noisy_testpoint_labels)
    axes.legend()
    fig.savefig(os.path.join(save_path, f'emission_distribution_{t}.png'))


    # Plot the full regression w/ verticle histograms
    fig, axes = plt.subplots(1)
    regression_mean_line_with_emission_distributions(
        axes, flattened_parameters, 
        reg_model, noise_model,
        data_loader.data[list(data_loader.training_indices)].numpy(),
        data_loader.labels[list(data_loader.training_indices)].numpy(),
        data_loader.data.min(), data_loader.data.max()
    )
    axes.scatter(
        data_loader.test_data.flatten().numpy(), data_loader.test_labels.flatten().numpy(), 
        marker = 'x', label = 'Test data', color = 'red', label = 'Test data'
    )
    axes.legend()
    fig.savefig(os.path.join(save_path, f'full_regression_plot_{t}.png'))

    # Plot performance over time
    fig, axes = plt.subplots(1)
    axes.plot(all_test_marginal_log_likelihoods)
    axes.set_xlabel('After x acquisition round')
    axes.set_xlabel('Test set marginal log likelihood')
    fig.savefig(os.path.join(save_path, f'test_set_MLL.png'))

    # Visualise MCMC (against the real distribution if simulated dataset)
    # Do this for each J pair of parameters
    fig, axes = plt.subplots(1, args.regression_J, figsize = (5*args.regression_J, 5))

    for j in range(args.regression_J):

        axes[j].set_xlabel(f'$\logA_{j+1}$')
        axes[j].set_ylabel(f'$\log\\tau_{j+1}$')

        relevant_parameter_samples = parameter_samples[...,[j, j+args.regression_J]].numpy()    # [num steps, num chains, 2]

        for cc in range(num_chains):
            chain_samples = relevant_parameter_samples[:,cc,:]
            empirical_log_pis = np_parameter_unnormalised_log_posterior[:,cc]
            colours = plt.cm.plasma(empirical_log_pis)[:,:3]
            axes[j].scatter(*chain_samples.T, c = colours, marker = 'x')
            axes[j].plot(*chain_samples.T, c = 'grey', alpha = 0.2)

        x_minj, x_maxj = axes[j].get_xlim()
        y_minj, y_maxj = axes[j].get_ylim()

        x_min = min(x_min, x_minj)
        x_max = max(x_max, x_maxj)
        y_min = min(y_min, y_minj)
        y_max = max(y_max, y_maxj)

    # If simulated data, then also plot the true parameter distribution
    if dataset_type == 'simulated':

        assert args.generative_J == args.regression_J, "Need to think harder about how to plot the MCMC samples in \mathcal{H}!"

        x = torch.linspace(x_min, x_max, 100)
        y = torch.linspace(y_min, y_max, 100)
        xv, yv = torch.meshgrid(x, y, indexing='ij')
        gridded_h_space = torch.stack([xv for _ in range(args.generative_J)] + [yv for _ in range(args.generative_J)], -1)
        zv: _T = gen_hypothesis_space.prior.log_prob(gridded_h_space)   # [100, 100, dim H]

        for j in range(args.regression_J):
            xlimj = axes[j].get_xlim()
            ylimj = axes[j].get_ylim()
            zv_selected = zv[...,[j,  j+args.regression_J]].sum(-1).exp()
            axes[j].contour(xv.numpy(), yv.numpy(), zv_selected.numpy())
            axes[j].set_xlim(*xlimj)
            axes[j].set_ylim(*ylimj)

    axes[j].legend()
    title = 'MCMC chains. x colours = empirical posterior with labelled data.'
    if dataset_type == 'simulated':
        title = title + '\nContours = true hypothesis posterior used to generate the data'
    fig.suptitle(title)
    fig.savefig(os.path.join(save_path, f'MCMC_parameter_samples_{t}.png'))    
    ##########################################################################

    # Label next
    print(f'\tExpanding training dataset')
    data_loader.set_data_mode('query')
    data_loader.label(chosen_indices)
