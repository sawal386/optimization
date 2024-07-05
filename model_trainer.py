"""
contains methods and classes used in training( and testing) machine learning models
"""

import numpy as np
import torch
import copy

from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.utils import shuffle as shuffle_data

def sagd_update1(model, lr, beta, normalizer, z):
    """
    performs first round of updates of SAGD

    Args:
        model: (super class of nn.module) the model (one of models in models.py)
        lr: (float) learning rate
        beta: (float) the beta-parameter, the coupling parameter
        normalizer: (normalizer) float
        z: (tensor) the z-params for the model
    """

    with torch.no_grad():
        j = 0
        for p in model.parameters():
            y = p.data - lr * beta / normalizer * p.grad.data
            p.data = (1 - beta) * y + beta * z[j]
            j += 1
            

def sagd_update2(model, lr, normalizer, z):
    """
    performs the second update of SAGD
    
    Args:
        model: (super class of nn.Module) the model (one of models in models.py)
        lr: (float) learning rate
        normalizer: (normalizer) float
        z: (tensor) the z-params for the model 
    
    Returns:
        (tensor) the update values of z parameters 
    """

    with torch.no_grad():
        j = 0
        z_new = []
        for p in model.parameters():
            z_updated = z[j] - lr / normalizer * p.grad.data
            z_new.append(z_updated)
            j += 1

    return copy.deepcopy(z_new)


def predict_labels(y_hat):
    """
    predicts the labels

    Args:
        y_hat: (tensor) the predicted probability for each class
        
    Returns:
        (tensor) the predicted labels
    """

    pred_new = torch.ones(1, )
    if y_hat.dim() == 0:
        pred_new = torch.zeros((1, 2))
        pred_new[:, 0], pred_new[:, 1] = 1 - y_hat, y_hat
    elif y_hat.dim() == 1:
        pred_new = torch.zeros((y_hat.shape[0], 2))
        pred_new[:, 0], pred_new[:, 1] = 1-y_hat, y_hat
    labels = torch.argmax(pred_new, 1)

    return labels


def update_dict_list(dict_, **args):
    """
    updates the lists in a dictionary

    Args:
        dict_: (object) -> List[object]
         args: key-value pair
    """
    
    for keys in args:
        dict_[keys].append(args[keys])


def compute_normalizer(data, problem_type):
    """
    computes the normalizer to be used for SAGD
    
    Args:
        data: (tensor) the feature dataset
        problem_type: (str) regression or classification

    Returns: 
        (float) the normalizer
    """
    
    if problem_type == "regression":
        return torch.mean(torch.norm(data, dim=1) ** 2)
    elif problem_type == "classification":
        return torch.mean(torch.norm(data, dim=1) ** 2) / 2
    else:
        print("problem_type should be classication or regression. Returning 1")
        return 1


@torch.no_grad()
def param_reset(m):
    """
    resets the parameters of the model
    
    Args:
        m: (nn.module) the model 
    """

    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


class BaseModelTrainerTester:
    """
    Model trainer base type. used to train and test (supervised) models.
    
    Attributes:
        (int) n_epochs: total number of epochs for training
        (float) lr: the learning rate to be used for the model
        (str) optimizer_name: name of the optimizer to be used for the model
        (int) batch_size: the batch_size to be used for training
        (dict) train_test_dict: (str) -> (List(float)) dictionary used to store relevant
                              training-testing statistics
        ((torch.tensor, tensor)) train_data: tuple consisting of training features and their
                                                   corresponding labels
        ((torch.tensor, tensor)) test_data: tuple consisting of test features and their
                                                   corresponding labels
        (bool) shuffle: whether to shuffle the training data during training or not
        ([torch.tensor]) z_params: list of tensors; corresponding to z in SAGD
        (str) task_type: the nature of the task; can be either regression or classification
    """

    def __init__(self, n_epochs, lr, optimizer_name, batch_size, train_data,
                 task_type, test_data=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.train_test_dict = {"train_cost_iteration": [], "train_accuracy_iteration": [],
                                "train_cost_epoch": [], "train_accuracy_epoch": [],
                                "test_cost_epoch": [], "test_accuracy_epoch": []}
        self.train_data = train_data
        self.test_data = test_data
        self.task_type = task_type

    def compute_normalizer(self, data):
        return compute_normalizer(data, self.task_type)

    def get_train_cost_iteration(self):
        return self.train_test_dict["train_cost_iteration"]

    def get_train_accuracy_iteration(self):
        return self.train_test_dict["train_accuracy_iteration"]

    def get_train_cost_epoch(self):
        return self.train_test_dict["train_cost_epoch"]

    def get_train_accuracy_epoch(self):
        return self.train_test_dict["train_accuracy_epoch"]

    def get_test_cost_epoch(self):
        return self.train_test_dict["test_cost_epoch"]

    def get_test_accuracy_epoch(self):
        return self.train_test_dict["test_accuracy_epoch"]

    def eval_iteration_and_update(self, loss_func, y_prob, y_true, is_logistic, eval_correctness=False):
        """
        Computes the loss and updates the parameters
        xs
        Args:
            loss_func: (nn.Loss) the loss function that is in use
            y_prob: (tensor) the probability associated with each label
            y_true: (tensor) the true labels
            is_logistic: (tensor) whether the problem is logistic regression or not
            eval_correctness: (bool) whether to evaluate the correctness of the predictions or not

        Returns:
            (List[nn.Loss]) float
            (List[bool]) list indicating the correctness of the  predicted labels
        """

        if self.task_type == "regression":
            cost = loss_func(y_prob.flatten(), y_true) / 2
        else:
            if is_logistic:
                cost = loss_func(y_prob.reshape(y_true.shape), y_true.float())
            else:
                cost = loss_func(y_prob, y_true)
        if eval_correctness:
            with torch.no_grad():
                if is_logistic:
                    y_pred = predict_labels(y_prob.squeeze())
                else:
                    y_pred = predict_labels(y_prob)
                batch_correct = torch.eq(y_pred, y_true).tolist()
            return cost, batch_correct

        return cost


    def train_test_model(self, model, criterion, shuffle=True, logger=None, decay_lr=False,
                         beta=0.9, normalizer=1, is_logistic=False, compute_beta=False,
                         do_compute_normalizer=False, new_shape=None, decay_interval="batch",
                         dev="cpu", move_dev=False):
        """
        trains a given model and if provided with test data, also tests the model

        Args:
            model: (nn.Module) ML model implemented in models.py
            criterion: (nn.Loss) the loss function to be used
            decay_lr: (bool) whether to decay the learning rate or not
            logger: (logger)
            is_logistic: (bool) whether he problem is logistic regression or not
            beta: (float) the hyperparameter in Adam/SAGD
            normalizer: (float) the normalizer for SAGD
            shuffle: (bool) whether to shuffle the data or not
            compute_beta: (bool) whether to compute beta for SAGD or not
            do_compute_normalizer: (bool) whether to compute normalizer for SAGD or not
            new_shape: (tuple(int))shape of the data that is consistent with the model
            decay_interval: (int) the interval size over which we decay the learning rate
            dev : (str) the device (cuca/cpu/mps)
            move_dev: (bool) move to device or not (True is using CUDA/MPS)
        """

        train_indices = np.arange(0, self.train_data[0].shape[0] / self.batch_size).astype("int")
        if logger is None:
            print("No logger")
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        z_params = [p.clone().detach() for p in model.parameters()]

        if self.optimizer_name == "SAGD":
            if compute_beta:
                beta = (self.lr + 2 - np.sqrt(self.lr ** 2 +4)) / (2 * self.lr)

        elif self.optimizer_name == "Adam":
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, beta))
        elif self.optimizer_name == "Adagrad":
            opt = torch.optim.Adagrad(model.parameters(), lr=self.lr)
        elif self.optimizer_name == "SGD" or self.optimizer_name == "GD":
            opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer needs to be one of SAGD, Adam, Adagrad, SGD. Using SGD")
        print("Optimizer set to {}. Learning rate decay: {}".format(self.optimizer_name,
              decay_lr))

        decay_function = lambda epoch : 1 / ((epoch + 1) ** 0.5)
        scheduler = lr_scheduler.LambdaLR(opt, decay_function)
        train_X, train_y = self.train_data[0], self.train_data[1]
        permuted_X, permuted_y = self.train_data[0], self.train_data[1]

        for ep in tqdm(range(self.n_epochs)):
            ep_cost = 0
            ep_correct = []
            total_size = 0

            if shuffle:
                train_X, train_y = shuffle_data(permuted_X, permuted_y)

            for i_train in train_indices:
                batch_X = train_X[i_train*self.batch_size:(i_train+1)*self.batch_size]
                batch_y = train_y[i_train*self.batch_size:(i_train+1)*self.batch_size]
                if move_dev:
                    batch_X = batch_X.to(dev)
                    batch_y= batch_y.to(dev)
                if new_shape is not None:
                    batch_X.reshape((batch_X.shape[0],) + new_shape)
                total_size += batch_X.shape[0]
                if do_compute_normalizer:
                    normalizer = self.compute_normalizer(batch_X)
                opt.zero_grad()

                y_hat = model(batch_X)
                cost = 0
                if self.optimizer_name == "SAGD":
                    if self.task_type == "regression":
                        cost = self.eval_iteration_and_update(criterion, y_hat, batch_y, is_logistic)
                    elif self.task_type == "classification":
                        cost, batch_correct = self.eval_iteration_and_update(criterion, y_hat, batch_y, is_logistic,
                                                                             eval_correctness=True)
                        update_dict_list(self.train_test_dict, train_accuracy_iteration=np.mean(batch_correct))
                        ep_correct += batch_correct
                    ep_cost += cost.item() * self.batch_size
                    update_dict_list(self.train_test_dict, train_cost_iteration=cost.item())
                    cost.backward()
                    sagd_update1(model, self.lr, beta, normalizer, z_params)

                    opt.zero_grad()
                    y_hat1 = model(batch_X)
                    cost1 = self.eval_iteration_and_update(criterion, y_hat1, batch_y, is_logistic)
                    cost1.backward()
                    z_params = sagd_update2(model, self.lr, normalizer, z_params)
                else:
                    if self.task_type == "regression":
                        cost = self.eval_iteration_and_update(criterion, y_hat, batch_y, is_logistic)
                    else:
                        cost, batch_correct = self.eval_iteration_and_update(criterion, y_hat, batch_y, is_logistic,
                                                                 eval_correctness=True)
                        update_dict_list(self.train_test_dict, train_accuracy_iteration=np.mean(batch_correct))
                        ep_correct += batch_correct

                    update_dict_list(self.train_test_dict, train_cost_iteration=cost.item())
                    cost.backward()
                    if do_compute_normalizer and self.optimizer_name == "SGD":
                        with torch.no_grad():
                            for p in model.parameters():
                                p.grad.data = p.grad.data / normalizer
                    opt.step()
                    ep_cost += cost.item() * self.batch_size

                if decay_lr:
                    if decay_interval == "batch":
                        scheduler.step()

            update_dict_list(self.train_test_dict, train_cost_epoch=ep_cost/total_size)
            if self.task_type != "regression":
                update_dict_list(self.train_test_dict, train_accuracy_epoch=np.mean(ep_correct))

            if self.test_data is not None:
                pred_test_y = model(self.test_data[0])
                pred_labels_y = predict_labels(pred_test_y)
                accuracy = torch.eq(pred_labels_y, self.test_data[1]).float().mean().item()
                update_dict_list(self.train_test_dict, test_accuracy_epoch=accuracy)
                test_cost = criterion(pred_test_y, self.test_data[1])
                print("test results, accuracy:{}, cost:{}".format(accuracy, test_cost))
                update_dict_list(self.train_test_dict, test_cost_epoch=test_cost.item())

            if decay_lr:
                if decay_interval == "epoch":
                    scheduler.step()


class FunctionSequentialOptimizer:
    """
    class to optimize simple functions (with no labels) sequentially

    Attributes:
        (float) lr: learning rate to be used
        (str) optimizer_name: name of the optimizer to be used
        (str) optimal value: the optimal value of the function
        (nn.Module) model: the model whose parameters we are optimizing;
                           the argument for the forward method in the model
                           is a function
        (List[float]): the value of the function during the course of optimizatiom
        (List[float/np.ndarray/torch.tensor]): the values of the parameter during the course
                                               of optimization
        (torch.optim) opt: the optimizer in use
        (float) beta: the beta parameter
        (List[tensor]) z_params: the z-parameters for SAGD
        (int) total_steps: the total steps taken by the optimizer
    """

    def __init__(self, lr, model, optimizer_name, optimal_value=None, beta=0.9):
        self.lr = lr
        self.optimal_value = optimal_value
        self.function_values = []
        self.parameter_values = []
        self.model = model
        self.z_params = [p.clone().detach() for p in model.parameters()]

        self.beta = beta

        # initialize the optimizer

        if optimizer_name == "SAGD":
            self.beta = (self.lr + 2 - np.sqrt(self.lr ** 2 + 4)) / (2 * self.lr)
            self.opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif optimizer_name == "Adam":
            self.opt = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, self.beta))
        elif optimizer_name == "Adagrad":
            self.opt = torch.optim.Adagrad(model.parameters(), lr=self.lr)
        elif optimizer_name == "SGD" or optimizer_name == "GD":
            self.opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif optimizer_name == "NAGD":
            self.opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=beta,
                nesterov=True)
        self.optimizer_name = optimizer_name

        self.total_steps = 0

    def get_parameter_values(self):
        return np.array(self.parameter_values)

    def get_function_values(self):
        return np.array(self.function_values)

    def latest_function_value(self):
        return self.function_values[-1]

    def take_step(self, f_t, normalizer=1, decay_lr=False):
        """
        take a single step

        Args:
            f_t: (nn.Module) the function to optimize
            normalizer: (float/np.ndarray)  the normalizer to be used for the SAGD optimizer
            decay_lr: (bool) whether to decay the learning rate or not
        """

        self.total_steps += 1
        if decay_lr:
            print("learning rate decay:")
            self.opt.defaults["lr"] = self.lr / (self.total_steps + 1)
        self.opt.zero_grad()
        output = self.model(f_t)
        self.function_values.append(output.item())
        if self.model.get_parameter().dim() == 0:
            self.parameter_values.append(self.model.get_parameter().clone().detach().item())
        else:
            self.parameter_values.append(self.model.get_parameter().clone().detach())

        if self.optimizer_name == "SAGD":
            output.backward()
            sagd_update1(self.model, self.lr, self.beta, normalizer, self.z_params)
            self.opt.zero_grad()
            output1 = self.model(f_t)
            output1.backward()
            self.z_params = sagd_update2(self.model, self.lr, normalizer, self.z_params)
        else:
            output.backward()
            self.opt.step()

