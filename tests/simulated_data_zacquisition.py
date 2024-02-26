from tqdm import tqdm

import numpy as np

from tests.simulated_data_inference import mcmc_obj, data_loader, noise_model, reg_model
from acquisition_functions import RandomAcquisitionFunction, BALDMCAcquisitionFunction

acquisition_method = 'bald'
# acquisition_method = 'random'

if acquisition_method == 'bald':
    acquisition_function = BALDMCAcquisitionFunction(
        noise_model = noise_model, reg_model = reg_model, K_i = 16, 
    )

num_acquisition_rounds = 50
num_data_points_per_acquision_round = 5

num_steps = 256
num_chains = 4
initial_parameter_samples = None
initial_parameter_unnormalised_log_posterior = None

x_axis = np.linspace()


for t in tqdm(range(num_acquisition_rounds)):

    # Initialise or 'resume' sampling... this might not be optimal though!
    theta_sample_dict = mcmc_obj.sample_many(
        num_steps = num_steps, num_chains = num_chains if t == 0 else None,
        initial_hypotheses=initial_parameter_samples, 
        initial_unnormalised_log_posterior=initial_parameter_unnormalised_log_posterior
    )

    parameter_samples = theta_sample_dict['all_samples']                                          # [num_steps, num_chains, dim(H)]
    parameter_unnormalised_log_posterior = theta_sample_dict['all_unnormalised_log_posterior']    # [num_steps, num_chains]

    # Set up for next time
    initial_parameter_samples = parameter_samples[-1]
    initial_parameter_unnormalised_log_posterior = parameter_unnormalised_log_posterior[-1]


    # Get the acquisition score for each datapoint in the unlabelled set
    querying_set_scores, querying_set_indices = acquisition_function.evaluate_function_on_data_loader(
        data_loader = data_loader,
        all_parameter_estimates = parameter_samples.reshape(-1, mcmc_obj.hypothesis_space.dimensionality),
        use_tqdm=True
    )
    import pdb; pdb.set_trace()

    # Get the top <data_points_per_acquision_round> indices
    top_scoring_indices = querying_set_scores.argsort(descending = True)[:num_data_points_per_acquision_round]
    chosen_indices = querying_set_indices[top_scoring_indices]
    data_loader.label(chosen_indices)

    # XXX: some kind of evaluation!
    


