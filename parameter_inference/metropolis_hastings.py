from tqdm import tqdm

from typing import Union

import torch
from torch.distributions import Normal

from parameter_inference.mcmc_base import ParameterMCMCBase
from nonlinearregression.hypothesis_space import HypothesisSpaceBase as _H

class ParameterHomoskedasticGaussianMetropolisHastings(ParameterMCMCBase):

    def __init__(self, hypothesis_space, data_loader, reg_model, noise_model, kernel_std = 1.0) -> None:
        super().__init__(hypothesis_space, data_loader, reg_model, noise_model)
        dim = self.hypothesis_space.dimensionality
        self.proposal_kernel = Normal(loc=torch.zeros(dim).float(), scale=torch.ones(dim).float() * kernel_std)

    def propose(self, start_hypotheses):
        """
        start_hypotheses: [num_chains, dim(H)], and returns the same
        """
        num_chains = start_hypotheses.shape[0]
        kernel_prop = self.proposal_kernel.sample([num_chains])
        proposed = kernel_prop + start_hypotheses
        assert proposed in self.hypothesis_space        # so enforce [num_chains, dim(H)]
        return proposed

    def acceptance_probs(self, previous_hypotheses, proposed_hypotheses, initial_unnormalised_log_posterior=None):
        """
        previous_hypotheses, proposed_hypotheses of shape [num chains, dim(H)]
        initial_unnormalised_log_posterior (if given) of shape [num chains]

        acceptance_probabilities of shape [num chains] also, and things are super easy for the homoskedastic Gaussian case
        """
        if initial_unnormalised_log_posterior is None:
            initial_unnormalised_log_posterior = self.evaluate_unnormalised_log_posterior(previous_hypotheses)
        proposed_hypothesis_unnormalised_log_posterior = self.evaluate_unnormalised_log_posterior(proposed_hypotheses)

        acceptance_probabilities = (proposed_hypothesis_unnormalised_log_posterior - initial_unnormalised_log_posterior).exp().clip(max = 1.0)

        return {
            'initial_unnormalised_log_posterior': initial_unnormalised_log_posterior,  # [num chains]
            'proposed_hypothesis_unnormalised_log_posterior': proposed_hypothesis_unnormalised_log_posterior,  # [num chains]
            'acceptance_probabilities': acceptance_probabilities,   # [num chains]
        }

    def accept_or_reject(self, previous_hypotheses, proposed_hypotheses, initial_unnormalised_log_posterior, proposed_hypothesis_unnormalised_log_posterior, acceptance_probabilities):
        """
        Accepts or rejects on a per-chain basis.
        Because it may be used in the next proposal step, perform the same selection for the unnormalised_log_posteriors

        previous_hypotheses, proposed_hypotheses of shape [num chains, dim(H)]
        acceptance_probabilities of shape [num chains]
        initial_unnormalised_log_posterior, proposed_hypothesis_unnormalised_log_posterior of shape [num chains]
        """
        u_per_chain = torch.rand_like(acceptance_probabilities)
        accepted_indicators = u_per_chain <= acceptance_probabilities

        actual_next_hypotheses = previous_hypotheses.clone()
        actual_next_hypotheses[accepted_indicators] = proposed_hypotheses[accepted_indicators]

        actual_next_unnormalised_log_posterior = initial_unnormalised_log_posterior.clone()
        actual_next_unnormalised_log_posterior[accepted_indicators] = proposed_hypothesis_unnormalised_log_posterior[accepted_indicators]

        return {
            'actual_next_step': actual_next_hypotheses, # [num chains, dim(H)]
            'actual_next_unnormalised_log_posterior': actual_next_unnormalised_log_posterior, # [num chains]
            'accepted_indicators': accepted_indicators, # [num chains], used for logging...
        }

    def step_sample(self, initial_hypotheses=None, num_chains=None, initial_unnormalised_log_posterior=None):
        initial_hypotheses, num_chains = self.initialise_chains(initial_hypotheses, num_chains) # Checks that theta shaped [num_chains, dim(H)]

        # self.evaluate_unnormalised_log_posterior(thetas)
        proposed_next_step = self.propose(initial_hypotheses)

        # We may already have the unnormalised log posterior at that point, so no point reevaluating it!
        acceptance_probs_dict = self.acceptance_probs(initial_hypotheses, proposed_next_step, initial_unnormalised_log_posterior)

        actual_next_step_dict = self.accept_or_reject(
            previous_hypotheses = initial_hypotheses, 
            proposed_hypotheses = proposed_next_step, 
            initial_unnormalised_log_posterior = acceptance_probs_dict['initial_unnormalised_log_posterior'], 
            proposed_hypothesis_unnormalised_log_posterior = acceptance_probs_dict['proposed_hypothesis_unnormalised_log_posterior'], 
            acceptance_probabilities = acceptance_probs_dict['acceptance_probabilities']
        )

        return {
            'next_hypotheses': actual_next_step_dict['actual_next_step'],                                       # [num chains, dim(H)]
            'next_unnormalised_log_posterior': actual_next_step_dict['actual_next_unnormalised_log_posterior'], # [num chains]
            'accepted_indicators': actual_next_step_dict['accepted_indicators'],                                # [num chains], used for logging...
        }

    def sample_many(self, num_steps: int, initial_hypotheses = None, num_chains = None, initial_unnormalised_log_posterior = None):

        all_hypotheses = [initial_hypotheses]
        all_unnormalised_log_posterior = [initial_unnormalised_log_posterior]
        all_accepted_indicators = []

        for ns in tqdm(range(num_steps)):
            next_sample_info = self.step_sample(
                initial_hypotheses=all_hypotheses[-1],
                num_chains=num_chains if ns == 0 else None,
                initial_unnormalised_log_posterior=all_unnormalised_log_posterior[-1]
            )
            all_hypotheses.append(next_sample_info['next_hypotheses'])
            all_unnormalised_log_posterior.append(next_sample_info['next_unnormalised_log_posterior'])
            all_accepted_indicators.append(next_sample_info['accepted_indicators'])

        if initial_hypotheses is None:
            all_hypotheses = all_hypotheses[1:]
        if initial_unnormalised_log_posterior is None:
            all_unnormalised_log_posterior = all_unnormalised_log_posterior[1:]

        return {
            'all_samples': torch.stack(all_hypotheses, dim = 0),                                        # [num steps (/+1), num chains, dim(H)]
            'all_unnormalised_log_posterior': torch.stack(all_unnormalised_log_posterior, dim = 0),     # [num steps (//+1), num chains]
            'all_accepted_indicators': torch.stack(all_accepted_indicators, dim = 0),                   # [num steps, num chains]
        }

