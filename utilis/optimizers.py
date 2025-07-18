"""Advanced optimizers for K-Sparse AutoEncoder.

This module provides modern optimization algorithms including Adam, RMSprop,
and AdaGrad with support for sparse gradients and momentum.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from enum import Enum


class OptimizerType(Enum):
    """Enumeration of optimizer types."""
    SGD = "sgd"
    MOMENTUM = "momentum"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class BaseOptimizer(ABC):
    """Base class for optimizers."""
    
    def __init__(self, learning_rate: float = 0.01, **kwargs):
        """Initialize optimizer.
        
        Args:
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
        """
        self.learning_rate = learning_rate
        self.iterations = 0
        self.state = {}
    
    @abstractmethod
    def update(self, params: np.ndarray, gradients: np.ndarray, 
               param_name: str = "default") -> np.ndarray:
        """Update parameters using gradients.
        
        Args:
            params: Current parameters
            gradients: Parameter gradients
            param_name: Name of parameter (for state tracking)
            
        Returns:
            Updated parameters
        """
        pass
    
    def reset_state(self) -> None:
        """Reset optimizer state."""
        self.state = {}
        self.iterations = 0
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'iterations': self.iterations
        }


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, **kwargs):
        """Initialize SGD optimizer."""
        super().__init__(learning_rate, **kwargs)
    
    def update(self, params: np.ndarray, gradients: np.ndarray, 
               param_name: str = "default") -> np.ndarray:
        """Update parameters using basic SGD."""
        self.iterations += 1
        return params - self.learning_rate * gradients


class MomentumOptimizer(BaseOptimizer):
    """SGD with momentum optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, 
                 nesterov: bool = False, **kwargs):
        """Initialize momentum optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
    
    def update(self, params: np.ndarray, gradients: np.ndarray, 
               param_name: str = "default") -> np.ndarray:
        """Update parameters using momentum."""
        if param_name not in self.state:
            self.state[param_name] = np.zeros_like(params)
        
        velocity = self.state[param_name]
        
        # Update velocity
        velocity = self.momentum * velocity + self.learning_rate * gradients
        
        # Apply update
        if self.nesterov:
            # Nesterov momentum
            update = self.momentum * velocity + self.learning_rate * gradients
        else:
            # Standard momentum
            update = velocity
        
        self.state[param_name] = velocity
        self.iterations += 1
        
        return params - update
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        config = super().get_config()
        config.update({
            'momentum': self.momentum,
            'nesterov': self.nesterov
        })
        return config


class AdamOptimizer(BaseOptimizer):
    """Adam optimizer with bias correction."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, 
                 amsgrad: bool = False, **kwargs):
        """Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            epsilon: Small constant for numerical stability
            amsgrad: Whether to use AMSGrad variant
        """
        super().__init__(learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
    
    def update(self, params: np.ndarray, gradients: np.ndarray, 
               param_name: str = "default") -> np.ndarray:
        """Update parameters using Adam."""
        if param_name not in self.state:
            self.state[param_name] = {
                'm': np.zeros_like(params),  # First moment
                'v': np.zeros_like(params),  # Second moment
                'v_hat_max': np.zeros_like(params) if self.amsgrad else None  # Max v_hat for AMSGrad
            }
        
        state = self.state[param_name]
        m, v = state['m'], state['v']
        
        self.iterations += 1
        
        # Update moments
        m = self.beta1 * m + (1 - self.beta1) * gradients
        v = self.beta2 * v + (1 - self.beta2) * gradients**2
        
        # Bias correction
        m_hat = m / (1 - self.beta1**self.iterations)
        v_hat = v / (1 - self.beta2**self.iterations)
        
        # AMSGrad
        if self.amsgrad:
            v_hat_max = state['v_hat_max']
            v_hat_max = np.maximum(v_hat_max, v_hat)
            v_hat = v_hat_max
            state['v_hat_max'] = v_hat_max
        
        # Update parameters
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update state
        state['m'] = m
        state['v'] = v
        
        return params - update
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        config = super().get_config()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        })
        return config


class RMSpropOptimizer(BaseOptimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, 
                 epsilon: float = 1e-8, momentum: float = 0.0, 
                 centered: bool = False, **kwargs):
        """Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            rho: Decay rate for moving average
            epsilon: Small constant for numerical stability
            momentum: Momentum coefficient
            centered: Whether to center the second moment
        """
        super().__init__(learning_rate, **kwargs)
        self.rho = rho
        self.epsilon = epsilon
        self.momentum = momentum
        self.centered = centered
    
    def update(self, params: np.ndarray, gradients: np.ndarray, 
               param_name: str = "default") -> np.ndarray:
        """Update parameters using RMSprop."""
        if param_name not in self.state:
            self.state[param_name] = {
                'v': np.zeros_like(params),  # Moving average of squared gradients
                'mg': np.zeros_like(params) if self.centered else None,  # Moving average of gradients
                'momentum_buffer': np.zeros_like(params) if self.momentum > 0 else None
            }
        
        state = self.state[param_name]
        v = state['v']
        
        self.iterations += 1
        
        # Update moving averages
        v = self.rho * v + (1 - self.rho) * gradients**2
        
        if self.centered:
            mg = state['mg']
            mg = self.rho * mg + (1 - self.rho) * gradients
            denominator = np.sqrt(v - mg**2 + self.epsilon)
            state['mg'] = mg
        else:
            denominator = np.sqrt(v + self.epsilon)
        
        # Compute update
        update = self.learning_rate * gradients / denominator
        
        # Apply momentum if specified
        if self.momentum > 0:
            momentum_buffer = state['momentum_buffer']
            momentum_buffer = self.momentum * momentum_buffer + update
            update = momentum_buffer
            state['momentum_buffer'] = momentum_buffer
        
        # Update state
        state['v'] = v
        
        return params - update
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        config = super().get_config()
        config.update({
            'rho': self.rho,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'centered': self.centered
        })
        return config


class AdaGradOptimizer(BaseOptimizer):
    """AdaGrad optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8, 
                 lr_decay: float = 0.0, **kwargs):
        """Initialize AdaGrad optimizer.
        
        Args:
            learning_rate: Learning rate
            epsilon: Small constant for numerical stability
            lr_decay: Learning rate decay
        """
        super().__init__(learning_rate, **kwargs)
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.initial_learning_rate = learning_rate
    
    def update(self, params: np.ndarray, gradients: np.ndarray, 
               param_name: str = "default") -> np.ndarray:
        """Update parameters using AdaGrad."""
        if param_name not in self.state:
            self.state[param_name] = {
                'sum_squared_gradients': np.zeros_like(params)
            }
        
        state = self.state[param_name]
        sum_squared_gradients = state['sum_squared_gradients']
        
        self.iterations += 1
        
        # Apply learning rate decay
        if self.lr_decay > 0:
            self.learning_rate = self.initial_learning_rate / (1 + self.lr_decay * self.iterations)
        
        # Update sum of squared gradients
        sum_squared_gradients += gradients**2
        
        # Compute update
        update = self.learning_rate * gradients / (np.sqrt(sum_squared_gradients) + self.epsilon)
        
        # Update state
        state['sum_squared_gradients'] = sum_squared_gradients
        
        return params - update
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'lr_decay': self.lr_decay,
            'initial_learning_rate': self.initial_learning_rate
        })
        return config


class SparseAwareOptimizer(BaseOptimizer):
    """Sparse-aware optimizer wrapper for handling sparse gradients."""
    
    def __init__(self, base_optimizer: BaseOptimizer, 
                 sparsity_threshold: float = 1e-8, **kwargs):
        """Initialize sparse-aware optimizer.
        
        Args:
            base_optimizer: Base optimizer to wrap
            sparsity_threshold: Threshold for considering gradients sparse
        """
        super().__init__(base_optimizer.learning_rate, **kwargs)
        self.base_optimizer = base_optimizer
        self.sparsity_threshold = sparsity_threshold
    
    def update(self, params: np.ndarray, gradients: np.ndarray, 
               param_name: str = "default") -> np.ndarray:
        """Update parameters with sparse gradient handling."""
        # Identify non-sparse gradients
        non_sparse_mask = np.abs(gradients) > self.sparsity_threshold
        
        if np.any(non_sparse_mask):
            # Apply base optimizer only to non-sparse gradients
            sparse_gradients = gradients * non_sparse_mask
            updated_params = self.base_optimizer.update(params, sparse_gradients, param_name)
        else:
            # No significant gradients, return original parameters
            updated_params = params
        
        return updated_params
    
    def reset_state(self) -> None:
        """Reset optimizer state."""
        self.base_optimizer.reset_state()
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        config = self.base_optimizer.get_config()
        config.update({
            'sparsity_threshold': self.sparsity_threshold,
            'base_optimizer': type(self.base_optimizer).__name__
        })
        return config


class LearningRateScheduler:
    """Learning rate scheduler for optimizers."""
    
    def __init__(self, optimizer: BaseOptimizer, schedule_type: str = "constant", 
                 **schedule_params):
        """Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            schedule_type: Type of schedule (constant, step, exponential, cosine)
            **schedule_params: Schedule-specific parameters
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.schedule_params = schedule_params
        self.initial_lr = optimizer.learning_rate
        self.step_count = 0
    
    def step(self) -> None:
        """Update learning rate based on schedule."""
        self.step_count += 1
        
        if self.schedule_type == "constant":
            # No change
            pass
        elif self.schedule_type == "step":
            # Step decay
            step_size = self.schedule_params.get('step_size', 30)
            gamma = self.schedule_params.get('gamma', 0.1)
            
            if self.step_count % step_size == 0:
                self.optimizer.learning_rate *= gamma
        
        elif self.schedule_type == "exponential":
            # Exponential decay
            gamma = self.schedule_params.get('gamma', 0.99)
            self.optimizer.learning_rate = self.initial_lr * (gamma ** self.step_count)
        
        elif self.schedule_type == "cosine":
            # Cosine annealing
            T_max = self.schedule_params.get('T_max', 100)
            eta_min = self.schedule_params.get('eta_min', 0.0)
            
            self.optimizer.learning_rate = eta_min + (self.initial_lr - eta_min) * \
                                         (1 + np.cos(np.pi * self.step_count / T_max)) / 2
        
        elif self.schedule_type == "plateau":
            # Reduce on plateau (requires external loss monitoring)
            # This would typically be called from training loop with loss value
            pass
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.learning_rate


class OptimizerFactory:
    """Factory class for creating optimizers."""
    
    @staticmethod
    def create_optimizer(optimizer_type: OptimizerType, **kwargs) -> BaseOptimizer:
        """Create optimizer of specified type.
        
        Args:
            optimizer_type: Type of optimizer to create
            **kwargs: Optimizer parameters
            
        Returns:
            Configured optimizer instance
        """
        if optimizer_type == OptimizerType.SGD:
            return SGDOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.MOMENTUM:
            return MomentumOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.ADAM:
            return AdamOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.RMSPROP:
            return RMSpropOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.ADAGRAD:
            return AdaGradOptimizer(**kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_sparse_aware_optimizer(optimizer_type: OptimizerType, 
                                    sparsity_threshold: float = 1e-8, 
                                    **kwargs) -> SparseAwareOptimizer:
        """Create sparse-aware optimizer.
        
        Args:
            optimizer_type: Type of base optimizer
            sparsity_threshold: Sparsity threshold
            **kwargs: Base optimizer parameters
            
        Returns:
            Sparse-aware optimizer instance
        """
        base_optimizer = OptimizerFactory.create_optimizer(optimizer_type, **kwargs)
        return SparseAwareOptimizer(base_optimizer, sparsity_threshold)


# Convenience functions
def create_adam_optimizer(learning_rate: float = 0.001, **kwargs) -> AdamOptimizer:
    """Create Adam optimizer with default parameters."""
    return AdamOptimizer(learning_rate, **kwargs)


def create_rmsprop_optimizer(learning_rate: float = 0.001, **kwargs) -> RMSpropOptimizer:
    """Create RMSprop optimizer with default parameters."""
    return RMSpropOptimizer(learning_rate, **kwargs)


def create_momentum_optimizer(learning_rate: float = 0.01, momentum: float = 0.9, 
                            **kwargs) -> MomentumOptimizer:
    """Create momentum optimizer with default parameters."""
    return MomentumOptimizer(learning_rate, momentum, **kwargs)


def create_sparse_adam_optimizer(learning_rate: float = 0.001, 
                                sparsity_threshold: float = 1e-8, 
                                **kwargs) -> SparseAwareOptimizer:
    """Create sparse-aware Adam optimizer."""
    return OptimizerFactory.create_sparse_aware_optimizer(
        OptimizerType.ADAM, sparsity_threshold, learning_rate=learning_rate, **kwargs
    )