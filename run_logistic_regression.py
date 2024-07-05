"""
implements online logistic regression
"""
import argparse
import torch.nn as nn
import torch

from model_trainer import BaseModelTrainerTester
from models import LogisticRegression
from utils import *
from plotter import comparison_plot


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_data", type=int, help="number of data points")
    parser.add_argument("--n_dim", type=int, help="the dimension of the features")
    parser.add_argument("--optimizer", type=str, help="name of the optimizer")
    parser.add_argument("--bin_size", type=int, help="number of data points in each bin")
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
    print("Running Online Logistic Regression model. The optimizer "
          "is set to: {}".format(args.optimizer))

    log_reg_data = LogisticRegressionData(args.n_dim, args.n_data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = args.epochs
    if args.epochs == 1:
        shape = args.n_data

    all_costs = initialize_dict(args.learning_rates)
    all_accuracies = initialize_dict(args.learning_rates)

    for t in range(args.total_trials):
        print("trial #: {}".format(t))
        se = (t+1) * args.base_seed
        X, y = log_reg_data.generate_data(se, bin_size=args.bin_size,
                                          plot=False)
        for lr in args.learning_rates:
            model = LogisticRegression(args.n_dim, 1, se)
            trainer = BaseModelTrainerTester(args.epochs, lr, args.optimizer, args.batch_size,
                                             (X, y), "classification")
            if args.optimizer == "SAGD":
                trainer.train_test_model(model, nn.BCELoss(), shuffle=True, compute_beta=True,
                                         do_compute_normalizer=True, is_logistic=True)
            else:
                trainer.train_test_model(model, nn.BCELoss(), is_logistic=True,
                                         shuffle=True, beta=args.beta)
            all_costs[lr].append(trainer.get_train_cost_iteration())
            all_accuracies[lr].append(trainer.get_train_accuracy_iteration())

    costs_dir = Path(args.output_dir) / "train_costs"
    accuracies_dir = Path(args.output_dir) / "train_accuracies"
    figures_dir = Path(args.output_dir) / "figures"

    make_directories(costs_dir, figures_dir, accuracies_dir)
    suffix = ""
    if args.optimizer == "Adam" or args.optimizer == "SAGD":
        if args.beta != 0.9:
            suffix = "_" + str(args.beta)
    if args.decay_lr:
        suffix = suffix + "_decay"

    save_as_pickle(all_costs, costs_dir, "{}".format(args.optimizer+suffix))
    save_as_pickle(all_accuracies, accuracies_dir, "{}".format(args.optimizer + suffix))

    color_li = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black",
                "yellow"]
    color_dict_lr = map_two_lists(color_li, args.learning_rates)
    reformatted_costs = reformat_data(args.average_size_plot, all_costs)
    reformatted_accuracy = reformat_data(args.average_size_plot, all_accuracies)

    x_label = "#Data"
    if args.average_size_plot != 0:
        x_label = "#Data (averaged over {} data points)".format(args.average_size_plot)
        y_label = r"$f_t(x_t)$"
        y_limit = (1e-2, 1e3)
        comparison_plot(reformatted_costs, x_label, y_label, "{} performance, Online Logistic "
                        "Regression".format(args.optimizer), y_limit, color_dict_lr,
                        save_name=figures_dir/"{}".format(args.optimizer+suffix),
                        use_log=True, use_lim=True)
        comparison_plot(reformatted_accuracy, x_label, "Accuracy", "{} performance "
                        "Softmax Regression".format(args.optimizer), (1e-1,1), color_dict_lr,
                        save_name=figures_dir / "{}_accuracy".format(args.optimizer+suffix),
                        use_log=True, use_lim=True)
