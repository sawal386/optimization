"""
runs online linear regression on time-varying inputs
"""

import argparse
import torch.nn as nn
import torch

from model_trainer import BaseModelTrainerTester
from models import LinearRegression
from utils import *
from plotter import comparison_plot


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_data", type=int, help="number of data points")
    parser.add_argument("--n_dim", type=int, help="the dimension of the features")
    parser.add_argument("--optimizer", type=str, help="name of the optimizer")
    parser.add_argument("--total_trials", type=int, help="total number of trials to run")
    parser.add_argument("--base_seed", type=int, help="the seed value", default=97)
    parser.add_argument("--learning_rates", type=float, nargs="+",
                        help="the values of the learning rates")
    parser.add_argument("--batch_size", type=int, help="batch size to be used in training")
    parser.add_argument("--epochs", type=int, help="total number of epochs")
    parser.add_argument("--average_size_plot", type=int, default=0, help=
                        "#iterations over which average is computed")
    parser.add_argument("--beta", type=float, help="value of beta for SAGD / Adam",
                        default=0.9)
    parser.add_argument("--decay_learning_rate", dest="decay_lr", default=False,
                       action="store_true", help="whether or not to decay the learning rate")
    parser.add_argument("--output_dir", required=True, help="location where the output files"
                                                            "are saved")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print("Running Linear Regression model. The optimizer is is set to: {}".format(
        args.optimizer))

    lin_reg_data = LinearRegressionData(args.n_dim, args.n_data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = args.epochs
    if args.epochs == 1:
        shape = args.n_data

    all_costs = {lr: [] for lr in args.learning_rates}
    for t in range(args.total_trials):
        print("trial #: {}".format(t))
        se = (t+1) * args.base_seed
        X, y, w_true = lin_reg_data.generate_data(se)
        for lr in args.learning_rates:
            model = LinearRegression(args.n_dim, 1, se)
            trainer = BaseModelTrainerTester(args.epochs, lr, args.optimizer, args.batch_size,
                                             (X, y), "regression")
            if args.optimizer == "SAGD":
                trainer.train_test_model(model, nn.MSELoss(), shuffle=False, compute_beta=True,
                                         do_compute_normalizer=True)
            else:
                trainer.train_test_model(model, nn.MSELoss(), shuffle=False, beta=args.beta,
                                        decay_lr=args.decay_lr)
            all_costs[lr].append(trainer.get_train_cost_iteration())

    costs_dir = Path(args.output_dir) / "train_costs"
    figure_dir = Path(args.output_dir) / "figures"

    make_directories(costs_dir, figure_dir)
    suffix = ""
    if args.optimizer == "Adam":
        suffix = "_" + str(args.beta)

    save_as_pickle(all_costs, costs_dir, "{}".format(args.optimizer+suffix))

    color_li = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black"]
    color_dict_lr = map_two_lists(color_li, args.learning_rates)
    reformatted_costs = reformat_data(args.average_size_plot, all_costs)

    x_label = "#Data"
    if args.average_size_plot != 0:
        x_label = "#Data (averaged over {} data points)".format(args.average_size_plot)
        y_label = r"$f_t(x_t)$"
        y_limit = (1e-1, 1e1)
        comparison_plot(reformatted_costs, x_label, y_label,
                        "{} performance, Online Linear Regression".format(args.optimizer),
                        y_limit, color_dict_lr, save_name=figure_dir / "{}".format(args.optimizer+suffix),
                        use_log=True, use_lim=False)
