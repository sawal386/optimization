import pickle
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

def parse_arguments():

    parser = argparse.ArgumentParser(description="arguments needed for processing cifar 10 files")
    parser.add_argument("--base_loc", help='directory where the files are located')
    parser.add_argument("--file_names", help="names of the files", nargs="+")
    parser.add_argument("--output_loc", help="directory where the output files are saved")
    parser.add_argument("--output_name", help="name of the output file")

    return parser.parse_args()


def unpickle(file):
    """
    unpickles the cifar 10 dataset downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
    :param file: loc + name of the cifar10 file

    return: dict
    """
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def extract_features_and_labels(cifar_dict, label_key, data_key):
    """
    obtains the labels and features data from the unpickled cifar10 file

    :param (dict) cifar_dict: dictionary containing the cifar10 data and the labels
    :param (str) label_key: key with which the image labels are associated to
    :param (str) data_key: key with which the feature data are associated with

    return: (np.ndarray)
    """

    features = cifar_dict[data_key]
    labels = cifar_dict[label_key]

    return features, labels


if __name__ == "__main__":

    args = parse_arguments()
    all_features, all_labels = [], []
    for filename in args.file_names:
        dict_ = unpickle(Path(args.base_loc) / filename)
        features, labels = extract_features_and_labels(dict_, b"labels", b"data")
        all_features.append(features)
        all_labels = all_labels + labels

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels).reshape(-1, 1)
    feature_and_labels = np.hstack((all_labels, all_features))

    save_loc = Path(args.output_loc)
    save_loc.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(feature_and_labels)
    df.to_csv(save_loc / "{}.csv".format(args.output_name), index=False,
              header=False)

    df_mini = pd.DataFrame(feature_and_labels[0:5000])
    df_mini.to_csv(save_loc / "{}_mini.csv".format(args.output_name), index=False,
              header=False)
