"""K-sparse layer implementation for autoencoders.

This module provides a sparse layer that keeps only the k highest
activations and zeros out the rest, useful for sparse autoencoders.
"""
from typing import Callable
import numpy as np
from layers.linear_layer import LinearLayer
from utilis.activations import sigmoid_function


class SparseLayer(LinearLayer):
    """K-sparse linear layer.
    
    Inherits from LinearLayer but applies k-sparse constraint where only
    the k highest activations are kept per sample, rest are set to zero.
    
    Attributes:
        num_k_sparse: Number of activations to keep per sample
    """

    def __init__(self, name: str, n_in: int, n_out: int, activation: Callable[[np.ndarray, bool], np.ndarray] = sigmoid_function, num_k_sparse: int = 10) -> None:
        """Initialize k-sparse layer.
        
        Args:
            name: Layer name
            n_in: Number of input features
            n_out: Number of output features
            activation: Activation function to use
            num_k_sparse: Number of highest activations to keep
        """
        LinearLayer.__init__(self, name, n_in, n_out, activation)
        self.num_k_sparse = num_k_sparse

    def get_output(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with k-sparse constraint.
        
        Args:
            x: Input data (batch_size, n_in)
            
        Returns:
            K-sparse layer output (batch_size, n_out)
        """
        result = self.activation(x.dot(self.weights) + self.biases)

        k = self.num_k_sparse
        n_out = result.shape[1]
        
        # Apply sparsity constraint
        if k == 0:
            # Special case: zero out all activations
            result = np.zeros_like(result)
        elif k < n_out:
            # Vectorized implementation for better performance
            # Get indices of k largest elements for each sample
            indices = np.argpartition(result, -k, axis=1)[:, -k:]
            
            # Create mask and apply sparsity
            mask = np.zeros_like(result, dtype=bool)
            batch_indices = np.arange(result.shape[0])[:, np.newaxis]
            mask[batch_indices, indices] = True
            
            # Zero out non-selected elements
            result[~mask] = 0

        self.result = result
        return result
