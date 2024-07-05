"""
This file contains helper functions and classes
"""

from sklearn import datasets as dset
from abc import abstractmethod, ABC
from pathlib import Path

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def initialize_dict(key_values, value_type="list"):
    """
    Initializes an empty dictionary whose values are list

    Args:
        key_values: (object) the values of the keys
        value_type: the data type of the value associated with a key

    Returns:
         (dict) (object) -> (List[])
    """
    dict_ = {}
    for key in key_values:
        if value_type == "list":
            dict_[key] = []

    return dict_


class DataGenerator(ABC):
    """
    Data generation base type

    Attributes:
        (int) n_dim: dimension of the data
        (int) size: the number of data to
    """

    def __init__(self, n_dim, size):
        self.n_dim = n_dim
        self.size = size

    @abstractmethod
    def generate_data(self, seed, **others_args):
        """
        Abstract method for generating data.

        Args:
            seed: (int) number used to initialize random number generation
            others_args: (float) noise, (int) bin_size, (bool) plot

        Returns:
            (Tuple (torch.tensor)) synthetic data and other information
        """

        pass


class LinearRegressionData(DataGenerator):
    """
    Data for linear regression
    """
    def generate_data(self, seed, **other_args):
        """
        Returns:
             (torch.tensor) feature matrix
             (torch.tensor) labels
             (torch.tensor) coefficient
        """

        if "noise" in other_args:
            X, y, w = dset.make_regression(n_samples=self.size, n_features=self.n_dim,
                                           noise=other_args["noise"], n_informative=self.n_dim,
                                           coef=True, random_state=seed)
            return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(w)
        else:
            X, y, w = dset.make_regression(n_samples=self.size, n_features=self.n_dim,
                                           n_informative=self.n_dim, coef=True,
                                           random_state=seed * 11)
            scale = np.random.randint(1, 5, size = (self.size, 1))
            X = X * scale
            y = np.dot(X, w)

            return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(w)


def sigmoid(x):
    """
    The sigmoid function

    Args:
        x: (np.ndarray / float) value whose sigmoid is to be computed

    Returns:
        (np.ndarray / float) output of sigmoid
    """

    return 1 / (1 + np.exp(-x))


class LogisticRegressionData(DataGenerator):
    """
    Data for LogisticRegression
    """
    def generate_data(self, seed, **other_args):
        """
        Returns:
            (torch.tensor, ten)features and their corresponding labels(class)
        """

        bin_size = other_args["bin_size"]  # the number of data points in each bin
        plot = other_args["plot"]  # whether to plot the data
        np.random.seed(seed)
        data_X = np.zeros((self.size, self.n_dim))
        train_inds = np.arange(0, self.size / bin_size).astype(int)
        for i in train_inds[:-1]:
            b = (i+1)
            data_i = np.random.normal(0, b, (bin_size, self.n_dim))
            data_X[i*bin_size : (i+1)*bin_size] = data_i

        i = train_inds[-1]
        rem = self.size - i * bin_size
        high = (i+1)
        mid = high / 2
        data_X[i * bin_size : self.size] = np.random.normal(0, high, (rem, self.n_dim))
        w = np.random.normal(0, 1 / mid, self.n_dim)
        y = sigmoid(np.dot(data_X, w))

        if plot:
            plt.hist(y)
        y_lab = np.where(y < 0.5, 0, 1)
        #X, y = torch.from_numpy(data_X).float(), torch.from_numpy(y_lab).float()

        X, y = dset.make_classification(self.size, self.n_dim, n_informative=self.n_dim,
                                        n_redundant=0, n_clusters_per_class=1, flip_y=0,
                                        class_sep=1.0, scale=1)
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def make_directories(*paths):
    """
    Creates directories as per the path names

    Args:
        paths: (Path) paths of the directories
    """

    for path in paths:
        path.mkdir(exist_ok=True, parents=True)


def save_as_pickle(var, location, name):
    """
    Save data in pickle format

    Args:
        var: (object) the data structure / variable that is to be saved
        location: (Path) location where the data is to be saved
        name : (str) name of the saved file
    """

    with open(location / "{}.pkl".format(name), "wb") as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def map_two_lists(list1, list2):
    """
    Creates a map between the values in two lists. Needs to make sure that the two lists are equal

    Args:
        list1: (List[object]) list containing the values
        list2: (List[object]) list containing the keys

    Returns:
        (dict) map between the values in list 2 and list 1

    Raises:
        1. ValueError: If the two lengths are not equal
    """

    if len(list1) != len(list2):
        raise ValueError("The lengths of the two lists must be equal")

    return {list2[i]: list1[i] for i in range(len(list2))}


def reformat_data(average_size, data_dict):
    """
    Reduces the number of columns of a 2-dimensional by computing average over given range.
    For instance, if the data is of shape 5 x 100 and average size is 10. The resulting shape would be
    5 x 10. We can average over every 10 elements.

    Args:
        average_size: (int) the size over which average is computed
        data_dict: (dict) (object) -> (np.ndarray) the input data

    Returns:
        (dict) the data with reduced number of columns.
    """

    reduced_data_dict = {}
    if average_size == 0:
        for key in data_dict:
            reduced_data_dict[key] = np.array(data_dict[key])
    else:
        for key in data_dict:
            data = np.array(data_dict[key])
            actual_size = data.shape[1]
            q = actual_size // average_size
            #print(data.shape, actual_size, q, average_size)
            reshaped_data = np.array(data)[:, 0:q*average_size].reshape(
                len(data), q, average_size)
            reduced_data_dict[key] = np.mean(reshaped_data, axis=2)

    return reduced_data_dict


def load_dataset_csv(csv_file_loc, scale, device="cpu", reshape=False,
                     new_shape=None):
    """
    loads the data from a csv file

    Args:
        csv_file_loc: (Path) location of the csv file along with the name
        scale: (float) amount by which the datasets are scaled
        device: (str) the device
        reshape: (bool) whether to reshape the data or not
        new_shape: (tuple(int)) the shape to which the data is reshaped to

    Returns:
        (torch.tensor) features
         (torch.tensor) labels
    """

    print("load device:", device)
    whole_data = pd.read_csv(csv_file_loc, header=None).to_numpy()
    features = torch.tensor(whole_data[:, 1:]).float() * scale

    labels = torch.tensor(whole_data[:, 0]).to(device)

    if reshape:
        if new_shape is None:
            raise Exception("New shape needs to be provided")
        features = torch.reshape(features, (features.shape[0],) + new_shape)

    return features.to(device), labels


def select_best_lr(results_dict, use_log=False, type_="cost", ref_pos=0):
    """
    Select the best results based on a give metric

    Args:
        results_dict: (dict) (float) learning -> (np.ndarray) the associated costs
        use_log: (bool) whether to take logarithm of costs or not
        type_: (str) the metric used in evaluating the performances of learning rates; can be cost or  accuracy
        ref_pos: (int) the index after which data is taken into consideration

    Returns:
        (float) best performing lr
        (np.ndarray) results for best lr
    """

    best_lr = 0
    min_error = np.infty
    is_list = False
    for lr in results_dict:
        results = results_dict[lr]
        if isinstance(results, list):
            is_list = True
            results = np.array(results)
        results = results[:, ref_pos:]
        if type_ == "accuracy":
            results = 1 - results
        if use_log:
            error_mat = np.log10(results + 1)
        else:
            error_mat = results
        #error_mat[error_mat == np.infty] = np.nan
        avg_error = np.nanmean(error_mat, axis=0)
        lr_error = np.nanmean(avg_error)

        if lr_error < min_error:
            best_lr = lr
            min_error = lr_error
    best_results = np.array(results_dict[best_lr]) if is_list else results_dict[best_lr]
    if type_ == "cost":
        return best_lr, best_results
    else:
        return best_lr, 1 - best_results


def select_data(data_dict, opt_name):
    """
    Select data associated with a given optimizer

    Args:
        data_dict: (dict) (float/str) the learning rate -> (dict) (str) optimizer name ->  (np.ndarray) results
                   associated with the optimizer
        opt_name: (str) the name of the optimizer

    Returns:
        (dict) (str) optimizer name -> (dict) (float) learning rate -> (np.ndarray) the results
    """

    opt_dict = {}
    for k1 in data_dict:
        data_opt_k1_dict = data_dict[k1][opt_name]
        for k2 in data_opt_k1_dict:
            if k2 not in opt_dict:
                opt_dict[k2] = np.array(len(data_dict),
                                        data_opt_k1_dict[k2].shape[0])
            opt_dict[k2][k1] = data_opt_k1_dict[k2]

    return opt_dict
