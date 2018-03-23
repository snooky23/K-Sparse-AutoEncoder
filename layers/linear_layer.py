# Matrix and vector computation package
from utilis.activations import *


class LinearLayer:
    """The linear layer performs a linear transformation to its input."""

    def __init__(self, name, n_in, n_out, activation=sigmoid_function):
        self.name = name
        self.activation = activation
        self.result = []

        self.weights = 2 * np.random.random((n_in, n_out)) - 1
        self.biases = np.zeros(n_out)

    def get_output(self, x):
        result = self.activation(x.dot(self.weights) + self.biases)
        self.result = result
        return result
