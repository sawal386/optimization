"""
run_deep_network.py

implements a AlexNet like network to classify CIFAR10 images
"""

import argparse
import torch.nn as nn
import torch

from model_trainer import BaseModelTrainerTester
from models import AlexNet
from utils import *
from plotter import comparison_plot


def parse_arguments():
     parser = argparse.ArgumentParser()
     parser.add_argument("--optimizer", help="name of the optimizer")
     parser.add_argument("--total_trials", help="total number of trials of the experiments",
                         type=int)
     parser.add_argument("--base_seed", type=int, help="the seed value",
                         default=97)
     parser.add_argument("--learning_rates", type=float, nargs="+",
                         help="the values of the learning rates")
     parser.add_argument("--batch_size", type=int, default=128,
                         help="batch size to be used in training")
     parser.add_argument("--epochs", type=int, help="the total number of epochs"\
                         "used in training")
     parser.add_argument("--average_size_plot", type=int, default=0,
                         help="the number of iterations over which average is computed")
     parser.add_argument("--beta", type=float, help="value of beta fot SAGD/Adam",
                         default=0.999)
     parser.add_argument("--output_dir", help="location where the output is saved")
     parser.add_argument("--decay_learning_rate", dest="decay_lr", default=False,
                        action="store_true", help="whether or not to decay the learning rate")
     parser.add_argument("--train_data_loc", type=str, help="location of training dataset")
     parser.add_argument("--test_data_loc", type=str, help="location of test dataset")

     return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print("Running AlexNet on CIFAR 10 data")
    print("Decaying Learning Rate:{}".format(args.decay_lr))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    shape = args.epochs

    all_train_costs_epoch = initialize_dict(args.learning_rates)
    all_train_accuracies_epoch = initialize_dict(args.learning_rates)
    all_train_costs_iterations = initialize_dict(args.learning_rates)
    all_train_accuracies_iterations = initialize_dict(args.learning_rates)
    all_test_costs = initialize_dict(args.learning_rates)
    all_test_accuracies = initialize_dict(args.learning_rates)

    cifar_trainset = load_dataset_csv(args.train_data_loc, 1/255, device, True,
                                      (3, 32, 32))
    cifar_testset = load_dataset_csv(args.test_data_loc, 1/255, device, True,
                                     (3, 32, 32))

    for t in range(args.total_trials):
        print("trial #: {}".format(t))
        se = (t+1) * args.base_seed
        for lr in args.learning_rates:
            model = AlexNet(num_classes=10, seed=se).to(device)
            trainer = BaseModelTrainerTester(args.epochs, lr, args.optimizer,
                                             args.batch_size, cifar_trainset,
                                             "classification", test_data=cifar_testset)
            if args.optimizer == "SAGD":
                trainer.train_test_model(model, nn.CrossEntropyLoss().to(device),
                                        shuffle=True, compute_beta=False,
                                        do_compute_normalizer=False, is_logistic=False,
                                        beta=args.beta)
            else:
                trainer.train_test_model(model, nn.CrossEntropyLoss().to(device),
                                         is_logistic=False, beta=args.beta,
                                         decay_lr=args.decay_lr, shuffle=True)

            all_train_costs_iterations[lr].append(trainer.get_train_cost_iteration())
            all_train_accuracies_iterations[lr].append(trainer.get_train_accuracy_iteration())
            all_train_costs_epoch[lr].append(trainer.get_train_cost_epoch())
            all_train_accuracies_epoch[lr].append(trainer.get_train_accuracy_epoch())
            all_test_costs[lr].append(trainer.get_test_cost_epoch())
            all_test_accuracies[lr].append(trainer.get_test_accuracy_epoch())

    train_costs_iterations_dir = Path(args.output_dir) / "train_costs"
    train_accuracies_iterations_dir = Path(args.output_dir) / "train_accuracies"
    train_costs_epoch_dir = Path(args.output_dir) / "train_costs_epoch"
    train_accuracies_epoch_dir = Path(args.output_dir) / "train_accuracies_epoch"
    test_costs_dir = Path(args.output_dir) / "test_costs"
    test_accuracies_dir = Path(args.output_dir) / "test_accuracies"
    figures_dir = Path(args.output_dir) / "figures"

    make_directories(train_costs_iterations_dir, train_accuracies_iterations_dir,
                     train_costs_epoch_dir, train_accuracies_epoch_dir,
                     test_costs_dir, test_accuracies_dir, figures_dir)
    suffix = ""
    if args.optimizer == "Adam" or args.optimizer == "SAGD":
        suffix = "_" + str(args.beta)
    if args.decay_lr:
        suffix = suffix + "_decay"

    save_as_pickle(all_train_costs_iterations, train_costs_iterations_dir, "{}".format(
        args.optimizer+suffix))
    save_as_pickle(all_train_accuracies_iterations, train_accuracies_iterations_dir, "{}".format(
        args.optimizer+suffix))
    save_as_pickle(all_train_costs_epoch, train_costs_epoch_dir, "{}".format(
        args.optimizer+suffix))
    save_as_pickle(all_train_accuracies_epoch, train_accuracies_epoch_dir, "{}".format(
        args.optimizer+suffix))
    save_as_pickle(all_test_costs, test_costs_dir, "{}".format(
        args.optimizer+suffix))
    save_as_pickle(all_test_accuracies, test_accuracies_dir, "{}".format(
        args.optimizer+suffix))

    color_li = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black"]
    color_dict_lr = map_two_lists(color_li, args.learning_rates)

    reformatted_costs = reformat_data(args.average_size_plot, all_train_costs_epoch)
    reformatted_accuracy = reformat_data(args.average_size_plot, all_train_accuracies_epoch)

    x_label = "#Data"
    if args.average_size_plot != 0:
        x_label = "#Data (averaged over {} data points)".format(args.average_size_plot)
    y_label = r"$f_t(x_t)$"
    y_limit = (1e-2, 1e1)
    comparison_plot(reformatted_costs, x_label, y_label, "{} performance, AlexNet CIFAR10 "
                    "Network".format(args.optimizer), y_limit, color_dict_lr, save_name=figures_dir / "{}_cost".format(
                     args.optimizer+suffix), use_log=True, use_lim=True)

    comparison_plot(reformatted_accuracy, x_label, "Accuracy", "{} training accuracy, AlexNet CIFAR10 "
                    "Network".format(args.optimizer), (0.5, 1), color_dict_lr, save_name=figures_dir / "{}_accuracy".format(
                     args.optimizer+suffix), use_log=True, use_lim=True)
