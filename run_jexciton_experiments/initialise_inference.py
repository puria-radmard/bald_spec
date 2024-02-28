"""
All the classes needed for standard MCMC inference
At the moment, only the ParameterHomoskedasticGaussianMetropolisHastings class with a Normal prior is available
Generalising this will require a bit more clever argparse stuff...
"""


import torch
from torch.distributions import Normal


from hypothesis_space.hypothesis_space import HypothesisSpaceBase
from parameter_inference.metropolis_hastings import ParameterHomoskedasticGaussianMetropolisHastings

from hypothesis_space.hypothesis_space import HypothesisSpaceBase
from nonlinearregression.mean_fit import JExcitonModel
import noisemodels

from run_jexciton_experiments.initialise_dataset import data_loader, args


# Extract args
prior_exp_log_A_locs = args.prior_exp_log_A_locs
prior_exp_log_tau_locs = args.prior_exp_log_tau_locs
prior_log_A_scales = args.prior_log_A_scales
prior_log_tau_scales = args.prior_log_tau_scales
regression_J = args.regression_J
regression_noise_model_class_name = args.regression_noise_model_class_name
metropolis_hastings_kernel_std = args.metropolis_hastings_kernel_std
num_steps = args.num_steps
num_chains = args.num_chains


# Set up generative model
for param in [prior_exp_log_A_locs, prior_exp_log_tau_locs, prior_log_A_scales, prior_log_tau_scales]:
    assert len(param) >= regression_J
exp_log_A_loc = torch.tensor(prior_exp_log_A_locs[:regression_J])
exp_log_tau_loc = torch.tensor(prior_exp_log_tau_locs[:regression_J])
log_A_scale = torch.tensor(prior_log_A_scales[:regression_J])
log_tau_scale = torch.tensor(prior_log_tau_scales[:regression_J])
prior_distribution_loc = torch.concat([exp_log_A_loc, exp_log_tau_loc])
prior_distribution_scale = torch.concat([log_A_scale, log_tau_scale])
prior_distribution = Normal(loc = prior_distribution_loc.log(), scale = prior_distribution_scale)


# Set up regression model
reg_model = JExcitonModel(J=regression_J)
hypothesis_space = HypothesisSpaceBase(prior = prior_distribution, dimensionality=2*regression_J)
noise_model = getattr(noisemodels, regression_noise_model_class_name)()


# Set up MCMC sampler
mcmc_obj = ParameterHomoskedasticGaussianMetropolisHastings(
    hypothesis_space=hypothesis_space,
    data_loader=data_loader,
    reg_model=reg_model,
    noise_model=noise_model,
    kernel_std=metropolis_hastings_kernel_std
)
