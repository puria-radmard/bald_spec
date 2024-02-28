import argparse

from noisemodels import __all__ as available_noise_models


parser = argparse.ArgumentParser()

parser.add_argument("--save_path", type = str, required = True)

#################### START ACTIVE LEARNING ARGS ####################
parser.add_argument("--training_batch_size", type = int, required = False, default = 1000)  # Low comp!
parser.add_argument("--querying_batch_size", type = int, required = False, default = 64)    # High comp!
parser.add_argument("--starting_dataset_size", type = int, required = False, default = 5)   # Easy peasy
##################### END ACTIVE LEARNING ARGS #####################


#################### START DATA ARGS ####################
dataset_type_subparsers = parser.add_subparsers(help='select type of dataset', dest='dataset_type')   # Options = simulated, static

parser_a_simulated_data = dataset_type_subparsers.add_parser('simulated', help='options for the simulated data model geenerative model')
# parser_a_static_data = dataset_type_subparsers.add_parser('static', help='options for the static data model geenerative model')

# Arguments only required if using simulated data
# TODO: ADD OTHER OPTIONS BESIDES A NORMAL GENERATIVE DISTRIBUTION OVER (LOG)-HYPOTHESES
parser_a_simulated_data.add_argument("--sim_data_start", type = float, default = 0.1, required = False)
parser_a_simulated_data.add_argument("--sim_data_end", type = float, default = 1000, required = False)
parser_a_simulated_data.add_argument("--sim_data_count", type = int, default = 700, required = False)
parser_a_simulated_data.add_argument("--sim_test_data_count", type = int, default = 20, required = False)
parser_a_simulated_data.add_argument("--generative_J", type = int, default = 2, required = False)
parser_a_simulated_data.add_argument("--gen_exp_log_A_locs", type = float, nargs = '+', default = [1000.0, 800.0], required = False)
parser_a_simulated_data.add_argument("--gen_exp_log_tau_locs", type = float, nargs = '+', default = [150.0, 500.0], required = False)
parser_a_simulated_data.add_argument("--gen_log_A_scales", type = float, nargs = '+', default = [0.1, 0.1], required = False)
parser_a_simulated_data.add_argument("--gen_log_tau_scales", type = float, nargs = '+', default = [0.1, 0.1], required = False)
parser_a_simulated_data.add_argument("--generative_noise_model_class_name", type = str, default = 'UnitRateGammaNoiseModel', choices = available_noise_models, required = False)
parser_a_simulated_data.add_argument("--num_hypothesis", type = int, default = None, required = False)
##################### END DATA ARGS #####################



#################### START INFERENCE/MCMC ARGS ####################
# TODO: ADD OTHER OPTIONS BESIDES METROPOLIS HASTINGS AND NORMAL PRIOR
parser.add_argument("--prior_exp_log_A_locs", type = float, nargs = '+', default = [5000.0, 2500.0], required = False)
parser.add_argument("--prior_exp_log_tau_locs", type = float, nargs = '+', default = [100.0, 1000.0], required = False)
parser.add_argument("--prior_log_A_scales", type = float, nargs = '+', default = [0.3, 0.3], required = False)
parser.add_argument("--prior_log_tau_scales", type = float, nargs = '+', default = [0.3, 0.3], required = False)
parser.add_argument("--regression_J", type = int, default = 2, required = False)
parser.add_argument("--regression_noise_model_class_name", type = str, default = 'UnitRateGammaNoiseModel', choices = available_noise_models, required = False)
parser.add_argument("--metropolis_hastings_kernel_std", type = float, default = 0.05, required = False)
parser.add_argument("--num_steps", type = int, default = 512, required = False)
parser.add_argument("--num_chains", type = int, default = 1, required = False)
##################### END INFERENCE/MCMC ARGS #####################



parser.add_argument("--acquisition_method", type = str, choices = ['bald', 'random'], required = True)
parser.add_argument("--K_i", type = int, default = 16, required = False)
parser.add_argument("--num_acquisition_rounds", type = int, default = 50, required = False)
parser.add_argument("--num_data_points_per_acquision_round", type = int, default = 5, required = False)

args = parser.parse_args()
