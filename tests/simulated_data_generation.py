import torch
from torch.distributions import Normal

from data.simulated import SimulatedDataLoader

from nonlinearregression.hypothesis_space import HypothesisSpaceBase
from nonlinearregression.mean_fit import JExcitonModel
from noisemodels.base import UnitRateGammaNoiseModel

import matplotlib.pyplot as plt

# Data generation parameters
xs = torch.linspace(0.1, 1000, 1000).reshape(-1, 1) 
xs = (xs + torch.randn_like(xs) * 0.001).clip(min = 0.0)
num_hypothesis = 400

# What we will infer... but just simulate from for now
J = 1
true_distribution = Normal(loc = torch.tensor([1000.0, 200.0]).log(), scale = torch.tensor([0.1, 0.1]))
true_hypothesis_space = HypothesisSpaceBase(prior = true_distribution, dimensionality=2*J) # Ae^{-x/tau} where x is first then tau
reg_model = JExcitonModel(J=J)
noise_model = UnitRateGammaNoiseModel()

starting_dataset = 5    # Start with 5 datapoints

data_loader = SimulatedDataLoader(
    input_data = xs, 
    true_hypothesis_space=true_hypothesis_space,
    reg_model=reg_model,
    noise_model=noise_model,
    num_hypothesis=num_hypothesis,
    training_batch_size=1000,   # Easy peasy
    querying_batch_size=64, # Not so easy peasy...
    initial_labelled_dataset_size=starting_dataset,
)


if __name__ == '__main__':

    fig, [ax_data, ax_params] = plt.subplots(1, 2, figsize = (6, 3))

    x = data_loader.data.squeeze().numpy()
    y = data_loader.labels.squeeze().numpy()
    c = data_loader.selected_hypotheses.numpy() / num_hypothesis
    thetas = data_loader.thetas.numpy()
    ax_data.scatter(x, y, c=c, s=3)

    ax_params.scatter(*thetas.T, c = torch.arange(num_hypothesis).numpy() / num_hypothesis)
    ax_params.set_xlabel('log(A)')
    ax_params.set_ylabel('log(tau)')

    fig.savefig('tests/simulated_data.png')

