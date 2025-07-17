"""Activation functions for neural networks.

This module provides common activation functions and their derivatives
for use in neural network layers.
"""
from typing import Union
import numpy as np


def sigmoid_function(signal: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Sigmoid activation function.
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        
    Returns:
        Sigmoid activation or its derivative
    """
    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1.0 - signal)
    else:
        # Return the activation signal
        return 1.0 / (1.0 + np.exp(-signal))


def relu_function(signal: np.ndarray, derivative: bool = False) -> np.ndarray:
    """ReLU (Rectified Linear Unit) activation function.
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        
    Returns:
        ReLU activation or its derivative
    """
    if derivative:
        return (signal > 0).astype(float)
    else:
        # Return the activation signal
        return np.maximum(0, signal)


def tanh_function(signal: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Hyperbolic tangent activation function.
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        
    Returns:
        Tanh activation or its derivative
    """
    if derivative:
        # Return the partial derivation of the activation function
        return 1 - np.power(signal, 2)
    else:
        # Return the activation signal
        return np.tanh(signal)


def softmax_function(signal: np.ndarray) -> np.ndarray:
    """Softmax activation function.
    
    Args:
        signal: Input signal array
        
    Returns:
        Softmax probabilities
    """
    e_x = np.exp(signal - np.max(signal, axis=1, keepdims=True))
    signal = e_x / np.sum(e_x, axis=1, keepdims=True)
    return signal

# end activation function