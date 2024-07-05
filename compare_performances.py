"""
plots a figure comparing the performances of  optimization algorithms and
the variance of the performance over multiple runs
"""

import argparse
import matplotlib.pyplot as plt
import os

from utils import *
from plotter import plot_performance

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pkl_file_loc", help="location of the folder"
                        "containing the results (saved as pkl)")
    parser.add_argument("--files_list", default=None, nargs="+",
                        help="names of pkl files")
    parser.add_argument("--output_dir", required=True,
                        help="location of the output")
    parser.add_argument("--output_name", required=True, help="name of the "
                                                             "output file")
    parser.add_argument("--x_label", help="x axis label")
    parser.add_argument("--y_label_type", required=True, help="the type of y-label")
    parser.add_argument("--title", default="", help="title of the plot")
    parser.add_argument("--y_limit", help="y axis limits", nargs="+", type=float)
    parser.add_argument("--use_log", help="whether or not to plot the y-axis in log scale",
                        default=True, action="store_false")
    parser.add_argument("--average_size", type=int, help="number of iterations over which the "
                        "results are averaged", default=0)
    parser.add_argument("--plot_shades", default=True, action="store_false",
                        help="whether or not to plot shade corresponding to "\
                        "standard deviation or percentile")
    parser.add_argument("--take_log_error", default=False, action="store_true")
    parser.add_argument("--ref_position", default=0)
    parser.add_argument("--measure", help="the name of the measure", default=None)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    figure_dir = Path(args.output_dir)
    figure_dir.mkdir(exist_ok=True, parents=True)
    figname = args.output_name + ".pdf"
    name = figure_dir / figname

    files_list = []
    if args.files_list is not None:
        files_list = [Path(args.pkl_file_loc) / filename for filename in args.files_list]
    else:
        files_list = [Path(args.pkl_file_loc) / filename for filename in
                      os.listdir(args.pkl_file_loc)]

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    if args.measure is not None:
        fig = plt.figure(figsize=(13, 5))
        axes1 = fig.add_subplot(1, 2, 1)
        axes2 = fig.add_subplot(1, 2, 2)
        plot_performance(files_list, args.measure, args.title, args.x_label,
                    (1e-3, 10), args.average_size, True, args.plot_shades,
                    args.take_log_error, None, axes2, measure_type=args.measure)
        plot_performance(files_list, args.y_label_type, args.title, args.x_label,
                args.y_limit, args.average_size, args.use_log, args.plot_shades,
                args.take_log_error, None, axes1)
        fig.savefig(name, bbox_inches="tight")
    else:
        plot_performance(files_list, args.y_label_type, args.title, args.x_label,
                    args.y_limit, args.average_size, args.use_log, args.plot_shades,
                    args.take_log_error, None, axis, args.ref_position)

    fig.savefig(name, bbox_inches="tight")
    plt.close()