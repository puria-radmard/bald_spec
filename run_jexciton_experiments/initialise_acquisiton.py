"""
Generate the dataset for this experiment
Either simulated or static - i.e. all data is technically available from the start
An entirely new script will be needed when we move onto actually collecting data live over the course of active learning
"""

import torch

from run_jexciton_experiments.initialise_args import args
from run_jexciton_experiments.initialise_inference import noise_model, reg_model

from acquisition_functions import RandomAcquisitionFunction, BALDMCAcquisitionFunction



acquisition_method = args.acquisition_method
K_i = args.K_i
num_acquisition_rounds = args.num_acquisition_rounds
num_data_points_per_acquision_round = args.num_data_points_per_acquision_round

if acquisition_method == 'bald':
    acquisition_function = BALDMCAcquisitionFunction(
        noise_model = noise_model,
        reg_model = reg_model,
        K_i = K_i,
    )
elif acquisition_method == 'random':
    acquisition_function = RandomAcquisitionFunction()




