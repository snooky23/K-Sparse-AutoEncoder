"""Linear layer implementation for neural networks.

This module provides a fully connected linear layer with customizable
activation functions for use in neural network architectures.
"""
from typing import Callable
import numpy as np
from utilis.activations import sigmoid_function


class LinearLayer:
    """Fully connected linear layer.
    
    Performs linear transformation: output = activation(input @ weights + bias)
    
    Attributes:
        name: Layer name for identification
        activation: Activation function to apply
        weights: Weight matrix (n_in x n_out)
        biases: Bias vector (n_out,)
        result: Cached output from last forward pass
    """

    def __init__(self, name: str, n_in: int, n_out: int, activation: Callable[[np.ndarray, bool], np.ndarray] = sigmoid_function) -> None:
        """Initialize linear layer.
        
        Args:
            name: Layer name
            n_in: Number of input features
            n_out: Number of output features  
            activation: Activation function to use
        """
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.result: np.ndarray = np.array([])

        self.weights = 2 * np.random.random((n_in, n_out)) - 1
        self.biases = np.zeros(n_out)

    def get_output(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer.
        
        Args:
            x: Input data (batch_size, n_in)
            
        Returns:
            Layer output after activation (batch_size, n_out)
        """
        result = self.activation(x.dot(self.weights) + self.biases)
        self.result = result
        return result
