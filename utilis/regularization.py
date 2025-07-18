"""Regularization techniques for neural networks.

This module provides various regularization methods to prevent overfitting
and improve model generalization.
"""
import numpy as np
from typing import List, Optional


def l1_regularization(weights: List[np.ndarray], lambda_reg: float = 0.01) -> float:
    """Compute L1 regularization penalty.
    
    Args:
        weights: List of weight matrices from all layers
        lambda_reg: L1 regularization strength
        
    Returns:
        L1 penalty value
    """
    l1_penalty = 0.0
    for w in weights:
        l1_penalty += np.sum(np.abs(w))
    return lambda_reg * l1_penalty


def l2_regularization(weights: List[np.ndarray], lambda_reg: float = 0.01) -> float:
    """Compute L2 regularization penalty.
    
    Args:
        weights: List of weight matrices from all layers
        lambda_reg: L2 regularization strength
        
    Returns:
        L2 penalty value
    """
    l2_penalty = 0.0
    for w in weights:
        l2_penalty += np.sum(w ** 2)
    return lambda_reg * l2_penalty


def elastic_net_regularization(weights: List[np.ndarray], 
                             lambda_l1: float = 0.01, 
                             lambda_l2: float = 0.01) -> float:
    """Compute Elastic Net regularization penalty (L1 + L2).
    
    Args:
        weights: List of weight matrices from all layers
        lambda_l1: L1 regularization strength
        lambda_l2: L2 regularization strength
        
    Returns:
        Elastic Net penalty value
    """
    return (l1_regularization(weights, lambda_l1) + 
            l2_regularization(weights, lambda_l2))


def dropout_mask(shape: tuple, dropout_rate: float = 0.5, 
                training: bool = True) -> np.ndarray:
    """Generate dropout mask for a layer.
    
    Args:
        shape: Shape of the layer output
        dropout_rate: Fraction of units to drop (0.0 to 1.0)
        training: Whether in training mode
        
    Returns:
        Dropout mask (1s and 0s)
    """
    if not training or dropout_rate == 0.0:
        return np.ones(shape)
    
    # Generate random mask
    mask = np.random.binomial(1, 1 - dropout_rate, shape)
    
    # Scale by 1/(1-dropout_rate) to maintain expected output
    return mask / (1 - dropout_rate)


def batch_normalization(x: np.ndarray, gamma: Optional[np.ndarray] = None,
                       beta: Optional[np.ndarray] = None,
                       epsilon: float = 1e-8,
                       training: bool = True,
                       momentum: float = 0.9,
                       running_mean: Optional[np.ndarray] = None,
                       running_var: Optional[np.ndarray] = None) -> tuple:
    """Apply batch normalization.
    
    Args:
        x: Input data (batch_size, features)
        gamma: Scale parameter
        beta: Shift parameter
        epsilon: Small constant for numerical stability
        training: Whether in training mode
        momentum: Momentum for running statistics
        running_mean: Running mean for inference
        running_var: Running variance for inference
        
    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    if gamma is None:
        gamma = np.ones(x.shape[1])
    if beta is None:
        beta = np.zeros(x.shape[1])
    
    if training:
        # Compute batch statistics
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        
        # Normalize
        x_norm = (x - batch_mean) / np.sqrt(batch_var + epsilon)
        
        # Update running statistics
        if running_mean is not None:
            running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        else:
            running_mean = batch_mean
            
        if running_var is not None:
            running_var = momentum * running_var + (1 - momentum) * batch_var
        else:
            running_var = batch_var
            
    else:
        # Use running statistics for inference
        if running_mean is None or running_var is None:
            raise ValueError("Running statistics required for inference mode")
        
        x_norm = (x - running_mean) / np.sqrt(running_var + epsilon)
    
    # Scale and shift
    output = gamma * x_norm + beta
    
    return output, running_mean, running_var


def gradient_clipping(gradients: List[np.ndarray], 
                     max_norm: float = 1.0,
                     norm_type: str = "l2") -> List[np.ndarray]:
    """Apply gradient clipping to prevent exploding gradients.
    
    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed norm
        norm_type: Type of norm ("l1", "l2", or "inf")
        
    Returns:
        List of clipped gradients
    """
    if norm_type == "l1":
        total_norm = sum(np.sum(np.abs(g)) for g in gradients)
    elif norm_type == "l2":
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    elif norm_type == "inf":
        total_norm = max(np.max(np.abs(g)) for g in gradients)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
    
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        gradients = [g * clip_coef for g in gradients]
    
    return gradients


def weight_decay(weights: List[np.ndarray], 
                decay_rate: float = 0.0001) -> List[np.ndarray]:
    """Apply weight decay regularization.
    
    Args:
        weights: List of weight matrices
        decay_rate: Weight decay rate
        
    Returns:
        List of weight decay penalties
    """
    return [decay_rate * w for w in weights]


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = np.inf
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, val_loss: float, model_weights: List[np.ndarray]) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model_weights: Current model weights
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = [w.copy() for w in model_weights]
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            return True
            
        return False
    
    def get_best_weights(self) -> Optional[List[np.ndarray]]:
        """Get the best weights found during training.
        
        Returns:
            Best weights if available
        """
        return self.best_weights


class LearningRateScheduler:
    """Learning rate scheduling utilities."""
    
    @staticmethod
    def step_decay(initial_lr: float, drop_rate: float = 0.5, 
                  epochs_drop: int = 10, epoch: int = 0) -> float:
        """Step decay learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            drop_rate: Rate to drop learning rate
            epochs_drop: Number of epochs between drops
            epoch: Current epoch
            
        Returns:
            Adjusted learning rate
        """
        return initial_lr * (drop_rate ** (epoch // epochs_drop))
    
    @staticmethod
    def exponential_decay(initial_lr: float, decay_rate: float = 0.96, 
                         epoch: int = 0) -> float:
        """Exponential decay learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            decay_rate: Decay rate per epoch
            epoch: Current epoch
            
        Returns:
            Adjusted learning rate
        """
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr: float, T_max: int, 
                        eta_min: float = 0.0, epoch: int = 0) -> float:
        """Cosine annealing learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            T_max: Maximum number of epochs
            eta_min: Minimum learning rate
            epoch: Current epoch
            
        Returns:
            Adjusted learning rate
        """
        return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2