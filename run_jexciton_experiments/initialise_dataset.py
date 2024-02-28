"""
Generate the dataset for this experiment
Either simulated or static - i.e. all data is technically available from the start
An entirely new script will be needed when we move onto actually collecting data live over the course of active learning
"""

import torch
from torch.distributions import Normal

from data.simulated import SimulatedDataLoader

from run_jexciton_experiments.initialise_args import args

from hypothesis_space.hypothesis_space import HypothesisSpaceBase
from nonlinearregression.mean_fit import JExcitonModel
import noisemodels

training_batch_size = args.training_batch_size
querying_batch_size = args.querying_batch_size
starting_dataset_size = args.starting_dataset_size

dataset_type = args.dataset_type

# Not collecting data but instead generating data using a chosen hierarchical generative models

if dataset_type == 'simulated':

    # Parameters of the generative model generating the simulated data
    sim_data_start = args.sim_data_start
    sim_data_end = args.sim_data_end
    sim_data_count = args.sim_data_count
    sim_test_data_count = args.sim_test_data_count
    generative_J = args.generative_J
    gen_exp_log_A_locs = args.gen_exp_log_A_locs
    gen_exp_log_tau_locs = args.gen_exp_log_tau_locs
    gen_log_A_scales = args.gen_log_A_scales
    gen_log_tau_scales = args.gen_log_tau_scales
    generative_noise_model_class_name = args.generative_noise_model_class_name
    if args.num_hypothesis == None:
        args.num_hypothesis = args.sim_data_count + args.sim_test_data_count
    num_hypothesis = args.num_hypothesis


    xs = torch.linspace(sim_data_start, sim_data_end, sim_data_count).reshape(-1, 1)
    xs_test = sim_data_start + torch.rand(sim_test_data_count, 1) * (sim_data_end - sim_data_start)
    
    # Set up generative model
    for param in [gen_exp_log_A_locs, gen_exp_log_tau_locs, gen_log_A_scales, gen_log_tau_scales]:
        assert len(param) >= generative_J
    exp_log_A_loc = torch.tensor(gen_exp_log_A_locs[:generative_J])
    exp_log_tau_loc = torch.tensor(gen_exp_log_tau_locs[:generative_J])
    gen_log_A_scale = torch.tensor(gen_log_A_scales[:generative_J])
    gen_log_tau_scale = torch.tensor(gen_log_tau_scales[:generative_J])
    gen_distribution_loc = torch.concat([exp_log_A_loc, exp_log_tau_loc])
    gen_distribution_scale = torch.concat([gen_log_A_scale, gen_log_tau_scale])
    gen_distribution = Normal(loc = gen_distribution_loc.log(), scale = gen_distribution_scale)
    
    gen_hypothesis_space = HypothesisSpaceBase(prior = gen_distribution, dimensionality=2*generative_J) # "prior" parameter name is a misnomer!
    generative_reg_model = JExcitonModel(J=generative_J)
    generative_noise_model = getattr(noisemodels, generative_noise_model_class_name)()

    data_loader = SimulatedDataLoader(
        input_data = xs, 
        test_input_data = xs_test,
        gen_hypothesis_space=gen_hypothesis_space,
        reg_model=generative_reg_model,
        noise_model=generative_noise_model,
        num_hypothesis=num_hypothesis,
        training_batch_size=training_batch_size,
        querying_batch_size=querying_batch_size,
        initial_labelled_dataset_size=starting_dataset_size,
    )



