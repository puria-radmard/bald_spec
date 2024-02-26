from acquisition_functions.base import AcquisitionFunctionBase

from noisemodels.base import NoiseModelBase as _NM
from nonlinearregression.mean_fit import NonLinearRegressionModelBase as _NL

class BALDMCAcquisitionFunction(AcquisitionFunctionBase):
    "A Monte Carlo approximation to BALD for the scalar output case"

    def __init__(self, noise_model: _NM, reg_model: _NL, K_i: int) -> None:
        self.noise_model = noise_model
        self.reg_model = reg_model
        self.K_i = K_i

        print('Find out if entropies sum in output dim or what...')

    def evaluate_function_on_batch(self, databatch, all_parameter_estimates, *args, **kwargs):
        """
        Shapes in:
            databatch: [B, dim(out)]
            all_parameter_estimates: [I, dim(H)]
        """
        # Get likelihood parameterisation at each of the B testpoint - shape checks done here
        function_eval = self.reg_model.evaluate_f(databatch, all_parameter_estimates) # [B, I, dim(out)]

        # Monte Carlo in the output space at each testpoint - i.e. a sample from each ensemble member (one sample from H)
        # Then get their logprobs and follow the BALD first term to get the entropy of the full ensemble
        sampled_noisy_testpoint_labels = self.noise_model.sample_output(self.K_i, function_eval)   # [K_i, B, I, dim out]
        evaluated_loglikelihoods = self.noise_model.log_likelihood_many(sampled_noisy_testpoint_labels, function_eval)
        entropy_of_ensemble = - evaluated_loglikelihoods.exp().mean(0).log().mean(1)

        # Second term is much easier!
        mean_of_entropies = self.noise_model.ensemble_entropies(function_eval).sum(-1).mean(1)

        bald_scores = entropy_of_ensemble - mean_of_entropies

        return {
            'scores': bald_scores, 
            'sampled_noisy_testpoint_labels': sampled_noisy_testpoint_labels.moveaxis(1, 0) # Can be used for plotting, but the first axis needs to be batch size...
        }

         

        
        
