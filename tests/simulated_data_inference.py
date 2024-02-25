import torch
from torch.distributions import Normal

from nonlinearregression.hypothesis_space import HypothesisSpaceBase
from parameter_inference.metropolis_hastings import ParameterHomoskedasticGaussianMetropolisHastings

# Can reuse most of these objects!
from tests.simulated_data_generation import reg_model, noise_model, data_loader, J


prior = Normal(loc = torch.tensor([5000.0, 100.0]).log(), scale = torch.tensor([0.3, 0.3]))
hypothesis_space = HypothesisSpaceBase(prior = prior, dimensionality=2*J)

mcmc_obj = ParameterHomoskedasticGaussianMetropolisHastings(
    hypothesis_space=hypothesis_space,
    data_loader=data_loader,
    reg_model=reg_model,
    noise_model=noise_model,
)

next_theta_sample_dict = mcmc_obj.step_sample(num_chains = 32)
next_theta = next_theta_sample_dict['']

import pdb; pdb.set_trace()

pass
