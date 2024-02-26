import torch
from torch.distributions import Normal

import matplotlib.pyplot as plt

from nonlinearregression.hypothesis_space import HypothesisSpaceBase
from parameter_inference.metropolis_hastings import ParameterHomoskedasticGaussianMetropolisHastings

# Can reuse most of these objects!
from tests.simulated_data_generation import reg_model, noise_model, data_loader, J, true_distribution as DONOTUSE


prior = Normal(loc = torch.tensor([5000.0, 100.0]).log(), scale = torch.tensor([0.3, 0.3]))
hypothesis_space = HypothesisSpaceBase(prior = prior, dimensionality=2*J)

mcmc_obj = ParameterHomoskedasticGaussianMetropolisHastings(
    hypothesis_space=hypothesis_space,
    data_loader=data_loader,
    reg_model=reg_model,
    noise_model=noise_model,
    kernel_std=0.2
)


if __name__ == '__main__':

    num_steps = 1024
    num_chains = 32

    theta_sample_dict = mcmc_obj.sample_many(num_steps = num_steps, num_chains = num_chains)

    all_samples = theta_sample_dict['all_samples']                                          # [64, 32, 2]
    all_unnormalised_log_posterior = theta_sample_dict['all_unnormalised_log_posterior']    # [64, 32]
    all_accepted_indicators = theta_sample_dict['all_accepted_indicators']                  # [64, 32]



    #### Let's do some nice plotting...
    fig, [axes_estimated_logpi_evolution, axes_logpi_comparison_sample_scatter] = plt.subplots(1, 2, figsize = (20, 10))

    ### First, plot the estimated unnormalised log posterior evoluation as we take more samples with the M-H algorithm
    # We started at the prior, and we should diffuse towards the estimate of the true posterior...
    step_samples_unnormalised_log_posterior_means = []
    for i, step_samples_unnormalised_log_posterior in enumerate(all_unnormalised_log_posterior):
        x = [i for _ in step_samples_unnormalised_log_posterior]
        y = step_samples_unnormalised_log_posterior.numpy()
        axes_estimated_logpi_evolution.scatter(x, y, c = 'blue', s = 5)
        step_samples_unnormalised_log_posterior_means.append(y.mean())
    axes_estimated_logpi_evolution.plot(step_samples_unnormalised_log_posterior_means, c = 'red', label = 'Mean across chains')

    axes_estimated_logpi_evolution.set_xlabel('Timestep')
    axes_estimated_logpi_evolution.set_ylabel('$log\pi(\\theta|D)$ of samples across chains')
    axes_estimated_logpi_evolution.set_title('Empirical $log\pi(\\theta|D)$ at each Metropolis-Hastings step')
    axes_estimated_logpi_evolution.legend()


    ### Then, we can look at the actual samples made by the chain, and plot the actual distribution of the hypotheses on the same axis
    all_unnormalised_posterior = (all_unnormalised_log_posterior - all_unnormalised_log_posterior.max())# .exp()
    min_logpi = all_unnormalised_posterior.min()
    max_logpi = all_unnormalised_posterior.max()
    scale_logpi = max_logpi - min_logpi
    scaled_logpi = (all_unnormalised_posterior - min_logpi) / scale_logpi
    for j, chain in enumerate(all_samples.permute(1, 0, 2)):
        empirical_log_pis = scaled_logpi[:,j].numpy()
        colours = plt.cm.plasma(empirical_log_pis)[:,:3]
        axes_logpi_comparison_sample_scatter.scatter(*chain.T.numpy(), c = colours, marker = 'x')  # XXX: also colour xs by empirical log-pi
        axes_logpi_comparison_sample_scatter.plot(*chain.T.numpy(), c = 'grey', alpha = 0.2)  # XXX: also colour xs by empirical log-pi
        
    x_min, x_max = axes_logpi_comparison_sample_scatter.get_xlim()
    y_min, y_max = axes_logpi_comparison_sample_scatter.get_ylim()
    x = torch.linspace(x_min, x_max, 100)
    y = torch.linspace(y_min, y_max, 100)
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    zv = DONOTUSE.log_prob(torch.stack([xv, yv], -1)).sum(-1).exp()
    axes_logpi_comparison_sample_scatter.contour(xv.numpy(), yv.numpy(), zv.numpy())

    axes_logpi_comparison_sample_scatter.set_title('MH chains. x colours = empirical posterior. contours = true hypothesis posterior')

    fig.savefig('/Users/puriaradmard/Documents/GitHub/active_learning_for_spectrography/tests/test_images/simulated_data_inference_MH_example.png')

    import pdb; pdb.set_trace()
