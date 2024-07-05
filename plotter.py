"""
# contains functions that can be used to make figures
"""

import matplotlib.pyplot as plt
from utils import *
import numpy as np

def comparison_plot(data_dict, x_lab, y_lab, title, y_lim, color_dict,
                    save_name=None, draw_vertical_line=False, use_log=False,
                    use_lim=False, in_between=False, axes=None, make_legend=True):
    """
    make a plot to compare multiple results

    Args:
        data_dict: (dict)  (str) -> (np.array)
        x_lab: (str) x_axis label
        y_lab: (str) y_axis label
        title: (str) title of the plot
        y_lim : (Tuple(float)) the y-axis limits
        color_dict: (dict) (float/str) -> (str) color name
        save_name : (Path) name used to save the file
        draw_vertical_line: (bool) whether to draw a vertical line or not
        use_log : (bool) whether to set y-axis in log scale or not
        use_lim : (bool) whether to use y_lim as y-axis limits or not
        in_between: (bool) whether to plot shades or not
        axes: (Axes) axes to draw plot
        make_legend: (bool) whether to make the legend or not
    """

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

    axes.set_xlabel(x_lab)
    axes.set_ylabel(y_lab)
    axes.set_title(title)
    mid = 0

    for keys in data_dict:
        y = np.median(data_dict[keys], axis=0)
        lower = np.percentile(data_dict[keys], 2.5, axis=0)
        upper = np.percentile(data_dict[keys], 97.5, axis=0)
        #std = np.std(data_dict[keys], axis=0)
        #y = np.mean(data_dict[keys], axis=0)
        #lower = y - std
        #upper = y + std
        x = np.linspace(1, y.shape[0], y.shape[0])
        color = color_dict[keys]
        axes.plot(x, y, "-", label=keys, alpha=0.5, color=color)
        if in_between:
            axes.fill_between(x, lower, upper, color=color_dict[keys],
                              alpha=0.1)
        mid = y.shape[0] // 2
    if draw_vertical_line:
        axes.axvline(x=mid, color="black", alpha=0.8)
    if make_legend:
        axes.legend(loc="upper right")
    if use_log:
        axes.set_yscale("log")
    if use_lim:
        axes.set_ylim(y_lim)

    if save_name is not None:
        plt.savefig("{}.pdf".format(save_name), bbox_inches="tight")
        plt.close()


def select_best_hyper_param(opt_names, results_dict):
    """
    select the best set p=of hyperparameters (for now the learning rate)

    Args:
        opt_names: (List[str]) names of the optimizers
        results_dict: (dict)

    Returns:
        (dict) (float) -> (np.ndarray)
    """

    def group_names(keys, name_list, additional_key="decay"):
        """group the names in name_list into categories given by keys"""
        grouped_dict = {}
        for key in keys:
            grouped_dict[key] = []
            for name in name_list:
                if key in name:
                    if additional_key in name:
                        new_key = "{}_{}".format(key, additional_key)
                        if new_key not in grouped_dict:
                            grouped_dict[new_key] = []
                        grouped_dict[new_key].append(name)
                    else:
                        grouped_dict[key].append(name)

        return grouped_dict

    best_performing = {}
    grouped_opt = group_names(opt_names, list(results_dict.keys()))
    for opt in grouped_opt:
        dict_hyper_param = {opt_hyper_param: results_dict[opt_hyper_param] for
                           opt_hyper_param in grouped_opt[opt]}
        best_hyper_param, lowest_error = select_best_lr(dict_hyper_param)
        best_performing[best_hyper_param] = lowest_error

    return best_performing


def plot_performance(file_names, y_label_type, title, x_label, y_limit,  average_size,
                use_log_plot, in_between, use_log_error, name_with_dir=None,
                axis=None, index_start=0, measure_type="function_value"):
    """
    Plots the performances of algorithms.

    Args:
        file_names: (List[str]) names of the pickle files in which the results are saved.
        y_label_type: (str) y-axis label type
        x_label: (str) label of the x-axis
        y_limit: (tuple(float)) y-axis limits
        average_size: (int) number of data points to average over
        axis: (Axes) axes on which the plots are made.
        name_with_dir: (Path) name and directory of the output plot.
        use_log_plot: (bool) whether to plot in logarithmic scale or not.
        in_between: (bool) whether to plot the shades or not.
        title: (str)title of the plot.
        measure_type: (str)  metric of interest.
        index_start: (int) starting index for the plot.
        use_log_error: (bool) whether to take logs of errors while selecting the best hyperparameter.
    """
    def generate_measure(data_dict, type_):
        """compute the measure given bu type_ from results in data_dict"""

        measure_dict = {}
        print("measure type:", type_)
        if type_ == "function_value":
            return data_dict
        else:
            for key in data_dict:
                if measure_type == "mean":
                    measure_dict[key] = np.mean(data_dict[key], axis=0, keepdims=True)
                elif measure_type == "variance":
                    measure_dict[key] = np.var(data_dict[key], axis=0, keepdims=True)
                elif measure_type == "std":
                    measure_dict[key] = np.std(data_dict[key], axis=0, keepdims=True)

            return measure_dict

    all_lowest_error = {}
    all_best_lr = {}
    n_data = 0
    total_trials = 0
    optimizer_names = []

    metric = "cost"
    if "error" in y_label_type:
        metric = "accuracy"

    all_errors = {}
    for filename in file_names:
        with open(filename, "rb") as f:
            file = pickle.load(f)
            name = " ".join(str(filename).split("/")[-1].strip(".pkl").split("_"))
            optimizer_names.append(name)
            all_errors[name] = file
            best_lr, lowest_error = select_best_lr(file, use_log_error, metric,
                                                   index_start)
            all_best_lr[name] = best_lr
            all_lowest_error[name] = lowest_error
            if n_data == 0:
                total_trials, n_data = lowest_error.shape

    all_lowest_error = select_best_hyper_param(["Adam", "Adagrad", "SGD",
                                               "SAGD"], all_lowest_error)
    all_lowest_error = generate_measure(all_lowest_error, measure_type)
    optimizer_names = list(all_lowest_error.keys())
    color_li = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                "black", "yellow", "navy"]
    color_dict = map_two_lists(color_li, optimizer_names)

    y_label = ""
    if y_label_type == "diff":
        y_label = r"$f_t(x_t) - f_t(x^*)$"
    elif y_label_type == "func":
        y_label = r"$f_t(x_t)$"
    else:
        y_label = y_label_type

    if average_size != 0:
        rem = n_data % average_size
        q = n_data // average_size
        if rem != 0:
            print("Not computing average. Plotting results for every iteration")
        else:
            if average_size != 1:
                x_label = "{} (averaged over {} points)".format(x_label, average_size)
            lowest_costs_avg = {}
            for key in all_lowest_error:
                reshaped_data = all_lowest_error[key].reshape(total_trials,
                                                              q, average_size)
                lowest_costs_avg[key] = np.mean(reshaped_data, axis=2)
            all_lowest_error = lowest_costs_avg
    use_lim = True if y_limit is not None else False
    comparison_plot(all_lowest_error, x_label, y_label, title, y_limit, color_dict,
                    save_name=name_with_dir, use_log=use_log_plot, use_lim=use_lim,
                    in_between=in_between, make_legend=True, axes=axis)
