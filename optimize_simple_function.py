"""
optimizes a simple time varying function of the form: f(x) = np.log (a exp(-x * b_t + x * b_t)).
We analyze two cases: 1) Change b_t randomly in each iteration 2) Switch the value of b_t
halfway through
"""

import argparse
import random
import numpy as np
import torch

from models import SimpleTimeVaryingModel
from model_trainer import FunctionSequentialOptimizer
from utils import *
from plotter import comparison_plot


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_data", required=True, type=int, help="number of data")
    parser.add_argument("--optimizer", help="name of the optimizer to be used")
    parser.add_argument("--generation_method", required=True, help="data generation method")
    parser.add_argument("--base_seed", type=int)
    parser.add_argument("--total_trials", type=int, help="total number of trials")
    parser.add_argument("--learning_rates", nargs='+', required=True, type=float,
                        help="learning rates")
    parser.add_argument("--output_dir", help="directory where the output is saved")
    parser.add_argument("--decay_lr", default=False, action="store_true",
                        help="whether or not to decay the learning rate")
    parser.add_argument("--beta", default=0.9, type=float, help="beta parameter for Adam")

    return parser.parse_args()


def generate_data(size, scheme, seed, a=1.0, b1=1.0, b2=7.0):
    """
    Generates data that serves as input to the function. 
    Args:
        size: (int) the total number of data points
        scheme: (str) the nature of the generated data
        seed: (int) the number used to initialize random number generation
        a: (float) constant parameter
        b1: (float) time-varying parameter
        b2: (float) time-varying parameter

    Returns: 
        (torch.tensor) values for a 
        (torch.tensor) values for b
    """

    A = [a for i in range(size)]
    B = [b1 for i in range(size)]
    if scheme == "step":
        B = [b1 if i < size/2 else b2 for i in range(size)]
    elif scheme == "random":
        random.seed(seed)
        B = [float(random.uniform(b1, b2)) for i in range(size)]
    elif scheme == "alternate":
        for i in range(size):
            B = [b1 if i % 2 == 0 else b2 for i in range(size)]

    return torch.tensor(np.array(A)), torch.tensor(np.array(B))


def evaluate_non_time_varying(size, learning_rates, optimizer_name, seed, b, beta):
    """
    Run the model when the incoming inputs are not time-varying

    Args:
        size: (int) the size of the dataset
        learning_rates: (List[float]) the learning rate values
        optimizer_name: (str) the name of the optimizer
        b: (float) the value of the constant parameter
        seed: (int) the number used to initialize random generations
        beta: (float) the beta value for Adam / SAGD

    Returns:
        (float) the best performing learning rate
    """

    data_a_all, data_b_all = generate_data(size, "constant", seed, b1=b)
    lowest_error = np.infty
    best_lr = 0
    np.random.seed(seed)
    start_x = random.uniform(5, 10)
    for lr in learning_rates:
        error = compute_sequential_parameters(start_x, lr, optimizer_name, data_a_all, data_b_all, beta)
        if np.nanmean(np.square(error)) < lowest_error:
            lowest_error = np.nanmean(np.square(error))
            best_lr = lr

    return best_lr


def compute_sequential_parameters(ini_point, learning_rate, optimizer_name, A, B, beta):
    """
    computes the value of the function when optimized sequentially

    Args:
        ini_point: (torch.tensor) starting position for the optimizer
        learning_rate: (float) the learning rate for the optimizer
        optimizer_name: (str) name of the optimizer to use
        A: (tensor) time varying parameters, b_t
        B: (tensor) time varying parameters, b_t

    Returns:
         (List[float]) the function value at every iteration
    """

    def optimize_model_sequentially(sequential_optimizer, a_vals, b_vals):
        """
        optimize a given model sequentially

        Args:
            sequential_optimizer: (FunctionSequentialOptimizer) the sequential optimizer
            a_vals: (tensor) values of the parameter a
            b_vals: (tensor) values of the parameter b

        Returns:
            (tensor) the objective function values during the course of optimization
        """

        total_data = a_vals.shape[0]
        for t in range(total_data):
            a_t, b_t = a_vals[t], b_vals[t]
            f_t = lambda x: torch.log(a_t) + torch.log(torch.exp(b_t * x) + torch.exp(b_t * (-x)))

            sequential_optimizer.take_step(f_t, normalizer=b_t**2)

    model = SimpleTimeVaryingModel(float(ini_point))
    model_optimizer = FunctionSequentialOptimizer(learning_rate, model, optimizer_name, beta=beta)

    optimize_model_sequentially(model_optimizer, A, B)

    return model_optimizer.get_function_values()


if __name__ == "__main__":
    args = parse_arguments()
    function_values = {lr:[] for lr in args.learning_rates}

    if args.generation_method == "step":
        best_lr = evaluate_non_time_varying(args.n_data // 2, args.learning_rates, args.optimizer,
                                            args.base_seed, 1.0, args.beta)
        function_values = {best_lr: []}
        data_a, data_b = generate_data(args.n_data, "step", args.base_seed, b1=1.0, b2=12.0)
        for lr in function_values:
            for t in range(args.total_trials):
                random.seed((t + 1) * args.base_seed)
                start = random.uniform(5, 10)
                function_values[lr].append(compute_sequential_parameters(start, lr, args.optimizer,
                                                                          data_a, data_b, args.beta))
    else:
        for t in range(args.total_trials):
            data_a, data_b = generate_data(args.n_data, args.generation_method,
                                           args.base_seed*(t+1), b1=1.0, b2=12.0)
            random.seed(( t+1) * args.base_seed)
            start = random.uniform(5, 10)
            for lr in args.learning_rates:
                function_values[lr].append(compute_sequential_parameters(start, lr, args.optimizer,
                                                                         data_a, data_b, args.beta))
    suffix = ""
    if args.optimizer == "Adam":
        suffix = "_" + str(args.beta)
    costs_dir = Path(args.output_dir + "_{}".format(args.generation_method)) / "cost"
    figure_dir = Path(args.output_dir + "_{}".format(args.generation_method)) / "figure"

    make_directories(costs_dir, figure_dir)
    save_as_pickle(function_values, costs_dir, "{}".format(args.optimizer + suffix))
    color_li = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black"]
    color_dict_lr = map_two_lists(color_li, args.learning_rates)

    x_label = "#Data"
    y_label = r"$f_t(x_t)$"
    y_limit = (0, 10)
    reformatted_costs = reformat_data(0, function_values)
    comparison_plot(reformatted_costs, x_label, y_label, "{} performance, simple {} "\
    "function".format(args.optimizer, args.generation_method), y_limit, color_dict_lr,
                    save_name=figure_dir / "{}".format(args.optimizer+suffix),
                    use_log=True, use_lim=False, make_legend=True)
