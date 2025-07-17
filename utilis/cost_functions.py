"""Cost functions for neural network training.

This module provides various cost/loss functions for training neural networks.
"""
import numpy as np


def subtract_err(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Simple difference error function.
    
    Args:
        outputs: Network predictions
        targets: True target values
        
    Returns:
        Difference between outputs and targets
    """
    res = outputs - targets
    return res


def mse(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean Squared Error cost function.
    
    Args:
        outputs: Network predictions
        targets: True target values
        
    Returns:
        Squared error between outputs and targets
    """
    res = np.power(outputs - targets, 2)
    return res


def cross_entropy_cost(outputs: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy cost function.
    
    Args:
        outputs: Network predictions (probabilities)
        targets: True target values (one-hot encoded)
        
    Returns:
        Cross-entropy loss value
    """
    epsilon = 1e-11
    targets = np.clip(targets, epsilon, 1 - epsilon)
    return -np.mean(outputs * np.log(targets) + (1 - outputs) * np.log(1 - targets))
