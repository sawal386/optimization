"""
# optimizers a simple quadratic function (positive definite) using SAGD and plots three parameters
# x, y, and z of the SAGD optimizer
"""

import argparse
import copy
import matplotlib.pyplot as plt
import torch

from utils import *
from pathlib import Path
from models import SimpleTimeVaryingModel
from model_trainer import FunctionSequentialOptimizer
import numpy as np

def parse_arguments():
    """
    parses the command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", default=1e-6, type=float,
                        help="threshold for stopping the optimizer")
    parser.add_argument("--output_dir", help="directory where the output is saved")
    parser.add_argument("--output_name", help="name of the output")
    parser.add_argument("--learning_rate", type=float, help="learning rate", dest="lr")
    parser.add_argument("--n_iterations", type=float, help="total number of iterations")
    parser.add_argument("--comparator_method", default=None)
    parser.add_argument("--use_log", default=True, action="store_false")

    return parser.parse_args()


def plot(data, axes, x_lab, y_lab, use_log=False, **kwargs):
    """
    Produces a plot of the data points

    Args:
        data: (dict) (str) optimizer name -> (np.ndarray) x and f(x)
        axes: (Axes) axes on which the plots are made
        x_lab: (str) the x-axis label
        y_lab: (str) the y-axis label
        use_log: (bool) whether to set the y-axis to log scale or not
    """

    axes.set_xlabel(x_lab)
    axes.set_ylabel(y_lab)

    if "starting_point" in  kwargs:
        axes.scatter(kwargs["starting_point"][0].item(),
                     kwargs["starting_point"][1].item(),
                     marker="x", label="starting", s=100)
    if "optimal_point" in kwargs:
        axes.scatter(kwargs["optimal_point"][0].item(),
                     kwargs["optimal_point"][1].item(),
                     marker="*", label="optimal", s=100)
    for keys in data:
        if "color_dict" in kwargs:
            try:
                axes.plot(data[keys][:, 0], data[keys][:, 1], "-o",
                      alpha=0.5, label=keys, markersize=2, color=kwargs["color_dict"][keys])
            except KeyError as e:
                axes.plot(data[keys][:, 0], data[keys][:, 1], "-o",
                          alpha=0.5, label=keys, markersize=2)
        else:
            axes.plot(data[keys][:, 0], data[keys][:, 1], "-o",
                      alpha=0.5, label=keys, markersize=2)

    axes.legend()
    if use_log:
        axes.set_yscale("log")
        #axes.set_xscale("symlog")

    return axes


def f(x):
    """
    evaluate the square of the L2 norm of the input vector

    Args:
         (tensor) x: input to the function

    Returns:
        (tensor) output
    """

    return torch.dot(x, x)


def booth(x):
    """
    Booth function
    Args:
        x: (tensor) input to the function

    Returns:
        (tensor) output
    """

    x1 = x[0]
    x2 = x[1]

    return (x1 + 2 * x2 -7)**2 + (2 * x1 + x2 - 5) ** 2


def matyas(x):
    """
    Matyas function
    Args:
        x: (tensor) input to the function

    Returns:
        (tensor) output
    """

    x1 = x[0]
    x2 = x[1]

    return (x1**2 + x2 ** 2) * 0.26 - 0.48*x1*x2


def rosenbrock(x):
    """
    Rosenbrock function

    Args:
        x: (tensor) input to the function

    Returns:
        (tensor) output
    """

    x1 = x[0]
    x2 = x[1]

    return (1 -x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

def perform_optimization_other(opt_name, x_start, func, alpha, total_iters, optimal_point,
                               tolerance=1e-6):
    """
    performs optimization for optimizers other than SAGD, NAGD, SGD
    Args:
        opt_name: (str) name of the optimizer
        x_start: (tensor) the initial parameter
        func:  (function) the function we are optimizing
        alpha: (float) the learning rate
        total_iters: (int) mthe total number of iterations
        optimal_point: (tensor) the optimal point
        tolerance: (tensor) the error tolerance

    Returns:
        (dict) (str) optimizer name -> (np.ndarray) parameter values
    """

    x = torch.nn.Parameter(x_start)
    x_params = [x]
    diff = func(x_params[0]) - func(optimal_point)

    all_x = [ini_x.numpy()]
    optimizer = torch.optim.SGD(x_params, lr=alpha)
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(x_params, lr=alpha)
    elif opt_name == "Adagrad":
        optimizer = torch.optim.Adagrad(x_params, lr=alpha)

    costs = []
    i = 0

    while i <= total_iters and diff > tolerance:
        optimizer.zero_grad()
        cost = func(x_params[0])
        costs.append(cost.item())
        cost.backward()
        all_x.append(x_params[0].detach().clone().numpy())
        optimizer.step()
        i += 1

    x_dict = {opt_name: np.array(all_x)[1:]}

    return x_dict


def merge_dict(dict1, dict2):
    """
    mergers two dictionaries and returns the merged dict
    """

    for key1 in dict1:
        if key1 not in dict2:
            dict2[key1] = dict1[key1]

    return dict2


def perform_optimization_SAGD(x_start, func, alpha, beta_sagd, total_iters,
                              normalizer, tolerance=1e-6):
    """
    optimizes a function using the SAGD optimizer

    Args:
        x_start: (tensor) the initial parameter value
        func: (function) the function we are optimizing
        alpha: (float) the learning rate
        beta_sagd: (float) the beta value for SAGD
        total_iters: (int) the total number of iterations
        normalizer: (float) normalizer for SAGD
        tolerance: (float) the tolerance value

    Returns:
        (dict) (str) SAGD variable name -> (np.ndarray) the parameter values
    """

    x = torch.nn.Parameter(x_start)
    x_params = [x]
    z_params = copy.deepcopy(x_params)
    diff = func(x_params[0]) - func(optimal)

    all_x = [ini_x.numpy()]
    all_z = [ini_x.numpy()]
    all_y = []

    optim_sagd = torch.optim.SGD(x_params, lr=alpha)
    costs = []
    i = 0

    while i <= total_iters and diff > tolerance:
        optim_sagd.zero_grad()
        cost = func(x_params[0])
        costs.append(cost.item())
        cost.backward()
        new_x = []
        with torch.no_grad():
            for j in range(len(x_params)):
                x = x_params[j]
                y = x.data - x.grad.data * alpha * beta_sagd / normalizer
                updated_x = (1 - beta) * y + z_params[j] * beta_sagd
                updated_x.requires_grad=True
                new_x.append(updated_x)
                all_y.append(y.detach().numpy())
                all_x.append(updated_x.detach().numpy())
            x_params = new_x

        optim_sagd.zero_grad()
        cost1 = func(x_params[0])
        cost1.backward()

        z_params_new = []
        with torch.no_grad():
            for k in range(len(z_params)):
                z_updated = z_params[k] - alpha / normalizer * x_params[k].grad.data
                z_params_new.append(z_updated)
                all_z.append(z_updated.detach().numpy())
            z_params = z_params_new
        diff = cost.item() - func(optimal).item()
        i += 1

    all_x, all_y, all_z = np.array(all_x), np.array(all_y), np.array(all_z)
    xyz_dict = {"x": all_x, "y": all_y, "z": all_z}

    return xyz_dict

def optimize_static_regressor(ini_point, learning_rate, optimizer_name,
                              func, momentum_param, normalizer, iters=100):
    """
    optimizes a static function using optimizers other than SAGD

    Args:
        ini_point: (tensor) the initial parameter value
        learning_rate: (float) the learning rate
        optimizer_name: (str) the optimizer to use
        func: (function) the function of interest
        momentum_param: (float) the momentum parameter
        normalizer: (float) the normalizer to use
        iters: (float) total number of iterations

    Returns:
        (np.ndarray) the function values over the course of optimization iterations

    """
    model = SimpleTimeVaryingModel(ini_point)
    model_optimizer = FunctionSequentialOptimizer(learning_rate, model,
                                                 optimizer_name, beta=momentum_param)
    t = 0
    while t < iters:
        model_optimizer.take_step(func, normalizer=normalizer)
        t += 1

    return model_optimizer.get_function_values()


if __name__ == "__main__":

    args = parse_arguments()

    ini_x = torch.tensor([18.0, 13.0])
    ini_x_clone = ini_x.clone().detach()
    N = 18 #the normalizer
    optimal = torch.tensor([1.0, 3.0])
    beta = (args.lr + 2 - np.sqrt(args.lr ** 2 + 4)) / (2 * args.lr)
    test_func = booth

    params_dict = perform_optimization_SAGD(ini_x.clone(), test_func, args.lr,
               beta, args.n_iterations, N)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(figsize=(11, 5))
    axes1 = fig.add_subplot(1, 2, 1)
    axes2 = fig.add_subplot(1, 2, 2)

    plot(params_dict, axes1, r"$x_1$", r"$x_2$", use_log=True,
         starting_point=ini_x_clone, optimal_point=optimal,
         color_dict={"x":"tab:red", "y":"tab:blue", "z":"tab:green"})
    optimizers = ["NAGD", "SAGD", "GD"]
    func_vals = {}
    for opt in optimizers:
        lr = 1 / 10
        if opt == "SAGD":
            lr = 1.5
        elif opt == "NAGD":
            lr = 1 / 18

        func_vals[opt] = optimize_static_regressor(ini_x_clone, lr, opt, test_func,
                                                  0.5, N, iters=60)
    func_vals_iter = {}
    for key in func_vals:
        values = func_vals[key]
        n = len(values)
        x_vals = np.arange(0, n)
        combined = np.hstack((x_vals.reshape(n, -1), values.reshape(n, -1)))
        func_vals_iter[key] = combined

    plot(func_vals_iter, axes2, "t, iterations", r"$f(x)$", use_log=True,
         color_dict={"NAGD": "tab:gray", "SAGD":"tab:purple", "GD":"tab:brown"})

    fig.savefig(out_dir / "{}.pdf".format(args.output_name), bbox_inches="tight")
