"""
implementations of machine learning models used in the paper
"""

import torch
import torch.nn as nn

class BasicModel(nn.Module):
    """
    Attributes:
        (int) n_dim: number of dimensions
        (int) n_output: the number of classes / final output dimensional
        (bool) include_bias: whether to include bias or not
        (nn.Linear) linear: a function that applies a linear transformation
    """
    def __init__(self, n_dim, n_output, seed, include_bias=True):

        super(BasicModel, self).__init__()
        torch.manual_seed(seed)

        self.n_dim = n_dim
        self.n_output = n_output
        self.include_bias = include_bias
        self.linear = nn.Linear(n_dim, n_output, bias=include_bias)

    def forward(self, x):
        """
        performs a single forward pass and returns the output

        Args:
            x: (tensor) input features

        Returns:
            (tensor) output of the forward pass
        """
        pass


class LinearRegression(BasicModel):
    """
    Implementation of linear regression model
    """
    def forward(self, x):

        return self.linear(x)


class LogisticRegression(BasicModel):
    """
    Implementation of logistic regression model
    """

    def forward(self, x):

        return torch.sigmoid(self.linear(x))


class SoftmaxRegression(BasicModel):
    """
    Implementation of the softmax regression model
    """
    def forward(self, x):

        return self.linear(x)


class FeedForwardNetwork(nn.Module):
    """
    Implementation of a simple feed forward neural network

    Attributes:
        (List[int]) dim_li: the number of hidden units in each layer
        (int) seed: the seed
        (bool) include_bias: whether to include bias or not
        (int) n_class: number of classes
        (List[nn.Linear]) fc_layers: list containing series of linear functions
        (nn.Linear) last_layer: a single linear layer
    """

    def __init__(self, n_dim, n_output, n_hidden_units_li, seed, include_bias=True):

        super(FeedForwardNetwork, self).__init__()
        self.dim_li = [n_dim] + [i for i in n_hidden_units_li]
        self.seed = seed
        self.include_bias = include_bias
        self.n_class = n_output
        self.n_hidden_units_li = n_hidden_units_li

        self.fc_layers, self.last_layer = self.initialize_net(self.seed)


    def initialize_net(self, seed):
        """
        initializes the layers of the feed forward network

        Args:
             seed: (int) number used in initializing random number generator

        Returns:
            (nn.ModuleList) the intermediate layers
            (nn.Linear) the last layer
        """

        fc_layers = []

        torch.manual_seed(seed)
        for i in range(1, len(self.dim_li)):
            n_inp = self.dim_li[i-1]
            n_out = self.dim_li[i]
            fc_layers.append(nn.Linear(n_inp, n_out, bias=self.include_bias))
        last_layer = nn.Linear(self.n_hidden_units_li[-1], self.n_class)
        fc_layer = nn.ModuleList(fc_layers)

        return fc_layer, last_layer


    def reset(self, seed):
        """
        resets the parameters of the network

        Args:
            seed: (int) number used in reinitializing the random number generator
        """

        self.fc_layers, self.last_layer = self.initialize_net(seed)


    def forward(self, x):
        """
        implements the forward propagation

        Args:
            x: (tensor) the input to the network

        Returns:
            (tensor) the output of forward prop
        """

        out = x
        for i in range(len(self.fc_layers)):
            out = torch.relu(self.fc_layers[i](out))
        out = self.last_layer(out)

        return out


class SimpleTimeVaryingModel(nn.Module):
    """
    Implementation of a model which consists of single parameter and no labels; the inputs to the model are time-varying.

    Attributes:
        (torch.tensor) x: parameters of the model
    """

    def __init__(self, init_value=None):
        """
        Args:
            init_value: (torch.tensor) the initial parameter
        """

        super(SimpleTimeVaryingModel, self).__init__()
        x = init_value
        if init_value is None:
            x = torch.distributions.Uniform(0, 5).sample()
        self.x = nn.Parameter(torch.tensor(x))

    def forward(self, f):
        """
        forward pass for the model; computing the function value

        Args:
            (lambda function) f : the function of interest

        Returns:
            (torch.tensor) output of the function
        """

        return f(self.x)

    def get_parameter(self):
        """
        Returns:
            (tensor) the model parameter
        """

        return self.x


class ConvNets(nn.Module):
    """
    Base class of deep networks consisting of convolutional networks

    Attributes:
        features: (nn.Sequential) the feature mapping network
        classifier: (nn.Sequential) the classification network
    """
    def __init__(self, num_classes=10, seed=97, batch_norm=True):
        """
        initializes the network

        Args:
            num_classes: (int) the number of classes
            seed: (int) number used to initialize random number generator
            batch_norm: (bool) whether to implement batch norm or not
        """
        super(ConvNets, self).__init__()
        torch.manual_seed(seed)
        self.num_classes = num_classes
        self.features = None
        self.classifier = None

    def forward(self, x):
        """
        implementation of the forward propagation step

        Args:
            x: (torch.tensor) the input tensor
        Returns:
            (torch.tensor) the output of forward prop
        """

        pass

class CifarNet(ConvNets):
    """
    implementation of CifarNet model described in AMS grad paper
    """
    def __init__(self, num_classes=10, seed=97, batch_norm=True):
        super(CifarNet, self).__init__(num_classes, seed, batch_norm)
        if batch_norm:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 64, kernel_size=6, stride=2, padding=2), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 64, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                )
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(64 * 2 * 2, 384), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(384, 192), nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
            )

    def forward(self, x):

        out = self.features(x)
        out = out.view(x.size(0), 64 * 2 * 2)
        out = self.classifier(out)

        return out

class AlexNet(ConvNets):
    """
    implementation of the AlexNet like model for CIFAR 10
    """

    def __init__(self, num_classes=10, seed=97, batch_norm=False):
        super(ConvNets, self).__init__(num_classes, seed, batch_norm)
        if batch_norm:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384),
                nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2), )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2), )
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 2 * 2, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)

        return x
