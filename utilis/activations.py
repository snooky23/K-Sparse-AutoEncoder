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


def leaky_relu_function(signal: np.ndarray, derivative: bool = False, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation function.
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        alpha: Negative slope coefficient
        
    Returns:
        Leaky ReLU activation or its derivative
    """
    if derivative:
        return np.where(signal > 0, 1.0, alpha)
    else:
        return np.where(signal > 0, signal, alpha * signal)


def elu_function(signal: np.ndarray, derivative: bool = False, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit (ELU) activation function.
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        alpha: Scale for negative values
        
    Returns:
        ELU activation or its derivative
    """
    if derivative:
        return np.where(signal > 0, 1.0, alpha * np.exp(signal))
    else:
        return np.where(signal > 0, signal, alpha * (np.exp(signal) - 1))


def swish_function(signal: np.ndarray, derivative: bool = False, beta: float = 1.0) -> np.ndarray:
    """Swish activation function (x * sigmoid(beta * x)).
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        beta: Scaling parameter
        
    Returns:
        Swish activation or its derivative
    """
    sigmoid_val = 1.0 / (1.0 + np.exp(-beta * signal))
    
    if derivative:
        return sigmoid_val + signal * sigmoid_val * (1 - sigmoid_val) * beta
    else:
        return signal * sigmoid_val


def gelu_function(signal: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function.
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        
    Returns:
        GELU activation or its derivative
    """
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = np.sqrt(2.0 / np.pi)
    
    if derivative:
        tanh_arg = sqrt_2_pi * (signal + 0.044715 * np.power(signal, 3))
        tanh_val = np.tanh(tanh_arg)
        sech_val = 1 - np.power(tanh_val, 2)
        
        return 0.5 * (1 + tanh_val) + 0.5 * signal * sech_val * sqrt_2_pi * (1 + 3 * 0.044715 * np.power(signal, 2))
    else:
        return 0.5 * signal * (1 + np.tanh(sqrt_2_pi * (signal + 0.044715 * np.power(signal, 3))))


def linear_function(signal: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Linear activation function (identity function).
    
    Args:
        signal: Input signal array
        derivative: If True, return derivative of function
        
    Returns:
        Linear activation or its derivative
    """
    if derivative:
        return np.ones_like(signal)
    else:
        return signal


# end activation function