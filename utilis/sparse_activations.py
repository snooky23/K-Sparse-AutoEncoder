"""Advanced sparse activation functions for K-Sparse AutoEncoders.

This module provides improved sparse activation functions including JumpReLU
with learnable thresholds and other advanced sparsity enforcement mechanisms.
"""
from typing import Tuple, Optional, Callable
import numpy as np
from enum import Enum


class SparseActivationType(Enum):
    """Enumeration of sparse activation types."""
    HARD_TOPK = "hard_topk"
    JUMP_RELU = "jump_relu"
    GATED_SPARSE = "gated_sparse"
    ADAPTIVE_SPARSE = "adaptive_sparse"


class BaseSparseActivation:
    """Base class for sparse activation functions."""
    
    def __init__(self, num_k_sparse: int, **kwargs):
        """Initialize sparse activation.
        
        Args:
            num_k_sparse: Number of neurons to keep active
            **kwargs: Additional configuration parameters
        """
        self.num_k_sparse = num_k_sparse
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sparse activation in forward pass.
        
        Args:
            x: Input activations
            
        Returns:
            Tuple of (sparse_output, sparsity_mask)
        """
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray, sparsity_mask: np.ndarray) -> np.ndarray:
        """Apply sparse activation in backward pass.
        
        Args:
            grad_output: Gradient from next layer
            sparsity_mask: Mask from forward pass
            
        Returns:
            Gradient for this layer
        """
        # Default: route gradients through mask
        return grad_output * sparsity_mask.astype(float)
    
    def update_parameters(self, learning_rate: float, **kwargs):
        """Update learnable parameters if any.
        
        Args:
            learning_rate: Learning rate for parameter updates
            **kwargs: Additional update parameters
        """
        pass


class HardTopKActivation(BaseSparseActivation):
    """Traditional hard top-k activation (current implementation)."""
    
    def __init__(self, num_k_sparse: int, **kwargs):
        """Initialize hard top-k activation."""
        super().__init__(num_k_sparse, **kwargs)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply hard top-k selection."""
        if self.num_k_sparse >= x.shape[1]:
            # No sparsity constraint needed
            mask = np.ones_like(x, dtype=bool)
            return x, mask
        
        # Get top-k indices for each sample
        indices = np.argpartition(x, -self.num_k_sparse, axis=1)[:, -self.num_k_sparse:]
        
        # Create mask
        mask = np.zeros_like(x, dtype=bool)
        batch_indices = np.arange(x.shape[0])[:, np.newaxis]
        mask[batch_indices, indices] = True
        
        # Apply mask
        result = x * mask.astype(float)
        return result, mask


class JumpReLUActivation(BaseSparseActivation):
    """JumpReLU activation with learnable thresholds for improved sparse learning."""
    
    def __init__(self, num_k_sparse: int, n_features: int, 
                 threshold_init: str = "data_driven", temperature: float = 1.0, 
                 **kwargs):
        """Initialize JumpReLU activation.
        
        Args:
            num_k_sparse: Number of neurons to keep active
            n_features: Number of features/neurons
            threshold_init: Initialization method for thresholds
            temperature: Temperature for soft thresholding
        """
        super().__init__(num_k_sparse, **kwargs)
        self.n_features = n_features
        self.temperature = temperature
        self.threshold_init = threshold_init
        
        # Initialize learnable thresholds
        if threshold_init == "zero":
            self.thresholds = np.zeros(n_features)
        elif threshold_init == "normal":
            self.thresholds = np.random.normal(0, 0.1, n_features)
        else:  # data_driven - will be set during first forward pass
            self.thresholds = np.zeros(n_features)
            self.initialized = False
        
        # For gradient computation
        self.threshold_gradients = np.zeros(n_features)
        self.last_input = None
        self.last_mask = None
    
    def _initialize_thresholds_data_driven(self, x: np.ndarray):
        """Initialize thresholds based on data statistics."""
        if hasattr(self, 'initialized') and not self.initialized:
            # Set thresholds to achieve approximately target sparsity
            target_activation_rate = self.num_k_sparse / self.n_features
            
            for i in range(self.n_features):
                # Set threshold so that target percentage of samples activate
                self.thresholds[i] = np.percentile(x[:, i], 
                                                 (1 - target_activation_rate) * 100)
            
            self.initialized = True
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply JumpReLU with learnable thresholds."""
        # Initialize thresholds if needed
        if self.threshold_init == "data_driven":
            self._initialize_thresholds_data_driven(x)
        
        # Store input for gradient computation
        self.last_input = x.copy()
        
        # Apply thresholds
        thresholded = np.maximum(0, x - self.thresholds)
        
        # Apply top-k constraint to maintain exact sparsity
        if self.num_k_sparse < x.shape[1]:
            # Get top-k indices based on thresholded values
            indices = np.argpartition(thresholded, -self.num_k_sparse, axis=1)[:, -self.num_k_sparse:]
            
            # Create mask
            mask = np.zeros_like(x, dtype=bool)
            batch_indices = np.arange(x.shape[0])[:, np.newaxis]
            mask[batch_indices, indices] = True
            
            # Apply mask
            result = thresholded * mask.astype(float)
        else:
            mask = thresholded > 0
            result = thresholded
        
        self.last_mask = mask
        return result, mask
    
    def backward(self, grad_output: np.ndarray, sparsity_mask: np.ndarray) -> np.ndarray:
        """Compute gradients for JumpReLU."""
        # Gradient w.r.t. input (using straight-through estimator)
        grad_input = grad_output * sparsity_mask.astype(float)
        
        # Gradient w.r.t. thresholds
        if self.last_input is not None and self.last_mask is not None:
            # Threshold gradients (negative because threshold subtracts from input)
            threshold_mask = (self.last_input > self.thresholds) & self.last_mask
            self.threshold_gradients = -np.sum(grad_output * threshold_mask.astype(float), axis=0)
        
        return grad_input
    
    def update_parameters(self, learning_rate: float, **kwargs):
        """Update learnable thresholds."""
        if hasattr(self, 'threshold_gradients'):
            self.thresholds -= learning_rate * self.threshold_gradients
            # Reset gradients
            self.threshold_gradients = np.zeros_like(self.threshold_gradients)


class GatedSparseActivation(BaseSparseActivation):
    """Gated sparse activation that separates magnitude and gating decisions."""
    
    def __init__(self, num_k_sparse: int, n_features: int, n_input: int, 
                 gate_init: str = "normal", **kwargs):
        """Initialize gated sparse activation.
        
        Args:
            num_k_sparse: Number of neurons to keep active
            n_features: Number of output features
            n_input: Number of input features
            gate_init: Initialization method for gate weights
        """
        super().__init__(num_k_sparse, **kwargs)
        self.n_features = n_features
        self.n_input = n_input
        
        # Initialize gate weights
        if gate_init == "normal":
            self.gate_weights = np.random.normal(0, 0.1, (n_input, n_features))
        else:  # zero
            self.gate_weights = np.zeros((n_input, n_features))
        
        self.gate_biases = np.zeros(n_features)
        
        # For gradient computation
        self.gate_weight_gradients = np.zeros_like(self.gate_weights)
        self.gate_bias_gradients = np.zeros_like(self.gate_biases)
        self.last_input = None
        self.last_gates = None
        self.last_mask = None
    
    def forward(self, x: np.ndarray, original_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply gated sparse activation.
        
        Args:
            x: Magnitude activations
            original_input: Original input for gate computation
            
        Returns:
            Tuple of (gated_output, sparsity_mask)
        """
        self.last_input = original_input.copy()
        
        # Compute gate logits
        gate_logits = original_input.dot(self.gate_weights) + self.gate_biases
        
        # Apply sigmoid to get gate probabilities
        gates = 1 / (1 + np.exp(-gate_logits))
        self.last_gates = gates
        
        # Apply top-k constraint on gates
        if self.num_k_sparse < x.shape[1]:
            indices = np.argpartition(gates, -self.num_k_sparse, axis=1)[:, -self.num_k_sparse:]
            
            mask = np.zeros_like(x, dtype=bool)
            batch_indices = np.arange(x.shape[0])[:, np.newaxis]
            mask[batch_indices, indices] = True
        else:
            mask = np.ones_like(x, dtype=bool)
        
        self.last_mask = mask
        
        # Apply magnitude ReLU
        magnitude = np.maximum(0, x)
        
        # Combine magnitude, gates, and mask
        result = magnitude * gates * mask.astype(float)
        
        return result, mask
    
    def backward(self, grad_output: np.ndarray, sparsity_mask: np.ndarray) -> np.ndarray:
        """Compute gradients for gated sparse activation."""
        if self.last_input is None or self.last_gates is None:
            return grad_output * sparsity_mask.astype(float)
        
        # Gradient w.r.t. magnitude (input)
        grad_magnitude = grad_output * self.last_gates * sparsity_mask.astype(float)
        
        # Gradient w.r.t. gate weights
        grad_gates = grad_output * sparsity_mask.astype(float)
        gate_sigmoid_grad = self.last_gates * (1 - self.last_gates)
        
        self.gate_weight_gradients = self.last_input.T.dot(grad_gates * gate_sigmoid_grad)
        self.gate_bias_gradients = np.sum(grad_gates * gate_sigmoid_grad, axis=0)
        
        return grad_magnitude
    
    def update_parameters(self, learning_rate: float, **kwargs):
        """Update gate parameters."""
        self.gate_weights -= learning_rate * self.gate_weight_gradients
        self.gate_biases -= learning_rate * self.gate_bias_gradients
        
        # Reset gradients
        self.gate_weight_gradients = np.zeros_like(self.gate_weight_gradients)
        self.gate_bias_gradients = np.zeros_like(self.gate_bias_gradients)


class AdaptiveSparseActivation(BaseSparseActivation):
    """Adaptive sparse activation that adjusts k based on input complexity."""
    
    def __init__(self, min_k: int, max_k: int, adaptation_method: str = "entropy", 
                 **kwargs):
        """Initialize adaptive sparse activation.
        
        Args:
            min_k: Minimum number of active neurons
            max_k: Maximum number of active neurons
            adaptation_method: Method for computing adaptive k
        """
        super().__init__((min_k + max_k) // 2, **kwargs)  # Default k
        self.min_k = min_k
        self.max_k = max_k
        self.adaptation_method = adaptation_method
    
    def _compute_adaptive_k(self, x: np.ndarray) -> np.ndarray:
        """Compute adaptive k values based on input complexity."""
        if self.adaptation_method == "entropy":
            # Normalize inputs to probabilities
            x_norm = np.abs(x) + 1e-8
            x_norm = x_norm / np.sum(x_norm, axis=1, keepdims=True)
            
            # Compute entropy
            entropy = -np.sum(x_norm * np.log(x_norm), axis=1)
            
            # Normalize entropy to [0, 1]
            max_entropy = np.log(x.shape[1])
            normalized_entropy = entropy / max_entropy
            
            # Map to k range
            adaptive_k = self.min_k + (self.max_k - self.min_k) * normalized_entropy
            
        elif self.adaptation_method == "variance":
            # Use variance as complexity measure
            variance = np.var(x, axis=1)
            normalized_variance = (variance - np.min(variance)) / (np.max(variance) - np.min(variance) + 1e-8)
            adaptive_k = self.min_k + (self.max_k - self.min_k) * normalized_variance
            
        else:  # "magnitude"
            # Use mean magnitude as complexity measure
            magnitude = np.mean(np.abs(x), axis=1)
            normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-8)
            adaptive_k = self.min_k + (self.max_k - self.min_k) * normalized_magnitude
        
        return np.clip(adaptive_k, self.min_k, self.max_k).astype(int)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adaptive sparse activation."""
        # Compute adaptive k for each sample
        adaptive_k_values = self._compute_adaptive_k(x)
        
        # Create mask for each sample with its own k
        mask = np.zeros_like(x, dtype=bool)
        result = np.zeros_like(x)
        
        for i, k in enumerate(adaptive_k_values):
            if k >= x.shape[1]:
                # No sparsity constraint
                mask[i] = True
                result[i] = x[i]
            else:
                # Apply top-k for this sample
                indices = np.argpartition(x[i], -k)[-k:]
                mask[i, indices] = True
                result[i] = x[i] * mask[i].astype(float)
        
        return result, mask


class SparseActivationFactory:
    """Factory class for creating sparse activation functions."""
    
    @staticmethod
    def create_sparse_activation(activation_type: SparseActivationType, 
                               num_k_sparse: int, **kwargs) -> BaseSparseActivation:
        """Create a sparse activation function of the specified type.
        
        Args:
            activation_type: Type of sparse activation to create
            num_k_sparse: Number of neurons to keep active
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured sparse activation instance
        """
        if activation_type == SparseActivationType.HARD_TOPK:
            return HardTopKActivation(num_k_sparse, **kwargs)
        elif activation_type == SparseActivationType.JUMP_RELU:
            return JumpReLUActivation(num_k_sparse, **kwargs)
        elif activation_type == SparseActivationType.GATED_SPARSE:
            return GatedSparseActivation(num_k_sparse, **kwargs)
        elif activation_type == SparseActivationType.ADAPTIVE_SPARSE:
            return AdaptiveSparseActivation(num_k_sparse, **kwargs)
        else:
            raise ValueError(f"Unknown sparse activation type: {activation_type}")