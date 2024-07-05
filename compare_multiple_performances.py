"""
extension of compare_performances.py. compare_multiple_performances.py plots on multiple axes.
Hence, the program can be used to plot multiple evaluation metrics.
"""

import argparse
import matplotlib.pyplot as plt
import os

from utils import *
from plotter import  plot_performance



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_locations", nargs="+", help="directories of folders "
                        "where the pickle files are saved")
    parser.add_argument("--pkl_files_name", nargs="+", help="names of the pkl files")
    parser.add_argument("--output_dir", required=True, help="location of the output")
    parser.add_argument("--output_name", required=True, help="name of the "
                                                             "output file")
    parser.add_argument("--x_label", help="x axis label")
    parser.add_argument("--y_label_types", required=True, help="the type of y-label",
                        nargs="+")
    parser.add_argument("--titles", default=None, help="title of the plot", nargs="+")
    parser.add_argument("--y_limit", help="y axis limits", nargs="+", type=float)
    parser.add_argument("--use_log", help="whether or not to plot the y-axis in log scale",
                        default=True, action="store_false")
    parser.add_argument("--plot_shades", default=True, action="store_false",
                        help="whether or not to plot shade corresponding to "\
                        "standard deviation or percentile")
    parser.add_argument("--average_size", type=int, help="number of iterations over which the "
                        "results are averaged", default=0)
    parser.add_argument("--take_log_error", default=False, action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    axes_shape_dict = {2:(1, 2), 3: (1, 3), 4:(2, 2), 1: (1, 1)}
    fig = plt.figure(figsize=(13, 10))
    n_axes = len(args.pkl_locations)
    axes_shape = axes_shape_dict[n_axes]
    axes_dict = {}
    count = 1
    print(args.plot_shades)

    figure_dir = Path(args.output_dir)
    figure_dir.mkdir(exist_ok=True, parents=True)
    figname = args.output_name + ".pdf"
    name = figure_dir / figname

    for i in range(1, axes_shape[0]+1):
        for j in range(1, axes_shape[1]+1):
            axes_dict[count-1] = fig.add_subplot(axes_shape[0], axes_shape[1], count)
            count += 1

    for i in range(len(args.pkl_locations)):
        files_list = []
        if args.pkl_files_name is not None:
            files_list = [Path(args.pkl_locations[i]) / filename for filename in args.pkl_files_name]
        else:
            files_list = [Path(args.pkl_locations[i]) / filename for filename in
                          os.listdir(args.pkl_locations[i])]

        plot_performance(files_list, args.y_label_types[i], args.titles[i], args.x_label,
                    (args.y_limit[2*i],args.y_limit[2*i+1] ), args.average_size, args.use_log,
                    args.plot_shades, args.take_log_error,None, axes_dict[i])
    fig.savefig(name)
