from tqdm import tqdm
import torch
import numpy as np
from torch import distributions as d
import  matplotlib.pyplot as plt

# Setup
fig, [ax_hists, ax_convergence] = plt.subplots(1, 2, figsize = (20, 10))
all_balds = []

# Set parameters for the various MCMC estimations
I = 128                                # Number of hypothese we sample
K_is = 2**torch.arange(4, 16)          # Number of individual output samples from each hypothesis, at EACH testpoint


# Emulate MCMC sampling from the posterior over parameters
hypothesis_posterior_at_testpoint = d.Normal(loc = torch.tensor([1000.0]), scale = torch.tensor([250.0]))    # i.e. at this x, f(x; \theta) has this distribution. Shape (B)
sampled_hypotheses = hypothesis_posterior_at_testpoint.sample([I])        # [I, B] where B = batchsize = 1 here

# For each of these hypotheses (i.e. means), we have a different distribution over y at test point x
data_liklihood_ensemble = d.Gamma(sampled_hypotheses, torch.ones_like(sampled_hypotheses))

for K_i in tqdm(K_is):

    sampled_noisy_testpoint_likelihoods = data_liklihood_ensemble.sample([K_i])         # [K_i, I, B]
    B = sampled_noisy_testpoint_likelihoods.shape[-1]

    # All of these are effectively sampled from q(y) (see README)
    # For each of these points, we have to evaluate q(y), which consts of a log(sum(p_i))
    evaluated_loglikelihoods = data_liklihood_ensemble.log_prob(sampled_noisy_testpoint_likelihoods.reshape(-1, 1, B))    # [K_i * I, I, B]

    # Follow equation to get this
    entropy_of_ensemble = - evaluated_loglikelihoods.exp().mean(1).log().mean(0)

    # Now the second term of BALD (much easer!)
    mean_of_entropies = data_liklihood_ensemble.entropy().mean(0)

    # Put it all together!
    bald_score = entropy_of_ensemble - mean_of_entropies
    all_balds.append(bald_score.item())

    # How good is the y-space MC at approximating q?
    num_bins = 50
    ax_hists.hist(sampled_noisy_testpoint_likelihoods.flatten().numpy(), num_bins, histtype=u'step', density = True, alpha = 0.2, label = K_i)


x_lim = ax_hists.get_xlim()
x_min = max(x_lim[0], 0.0)
test_ys = torch.linspace(x_min, x_lim[1], 100)
import pdb; pdb.set_trace()
test_pys = data_liklihood_ensemble.log_prob(test_ys).exp().mean(0)
ax_hists.plot(test_ys.numpy(), test_pys.numpy(), label = 'True ensemble prediction')
ax_hists.legend(title = f"$I = {I}, K_i = ?$")

ax_convergence.plot(K_is, all_balds)
ax_convergence.set_xlabel('K_i')
ax_convergence.set_ylabel('Test BALD score')
ax_convergence.set_xscale('log', base = 2)

fig.savefig('tests/ensemble_entropy_estimation.png')
