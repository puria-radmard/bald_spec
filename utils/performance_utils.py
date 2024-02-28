from tqdm import tqdm

import torch
from torch import Tensor as _T

from noisemodels.base import NoiseModelBase as _NM
from data.base import ActiveLearningDataLoaderBase as _DL
from nonlinearregression.mean_fit import NonLinearRegressionModelBase as _NR


def evaluate_test_marginal_log_likelihood(
    all_sampled_params: _T,
    reg_model: _NR,
    noise_model: _NM,
    data_loader: _DL
):
    """
    p(D_test | M)                                                   where M = model architecture
        = p(D_test | D_train, M)                                    due to i.i.d.
        = \int p(D_test, theta | D_train, M) dtheta                 due to marginalisation
        = \int p(D_test | theta, M) p(theta | D_train, M)) dtheta   due to chain rule and i.i.d. again
        approx = (p(D_test | theta_i, M)).mean(i wrt p(theta | D_train))

    Code self explanatory after that

    Shapes:
        all_sampled_params [I, dimH]
    """

    data_loader.set_data_mode('test')

    total_test_marginalised_log_likelihood = 0.0

    for test_batch in tqdm(data_loader, total = data_loader.num_batches):
        
        data = test_batch['data']                                   # [B, dim_in]
        labels = test_batch['labels']                               # [B, dim_out]

        test_function_eval = reg_model.evaluate_f(data, all_sampled_params)             # [B, I, dimH]
        test_log_likelihood = noise_model.log_likelihood(labels, test_function_eval)    # [B, I]

        test_marginalised_log_likelihood = test_log_likelihood.exp().mean(-1).log()
        test_marginalised_log_likelihood = test_marginalised_log_likelihood.nan_to_num(nan=None, neginf=0.0)
        total_test_marginalised_log_likelihood += test_marginalised_log_likelihood.sum()

    return total_test_marginalised_log_likelihood
