"""Improved sparse layer with configurable activation types and initialization methods.

This module provides an enhanced sparse layer that supports multiple sparsity
enforcement mechanisms, advanced initialization strategies, and improved training.
"""
from typing import Callable, Optional, Dict, Any
import numpy as np
from layers.linear_layer import LinearLayer
from utilis.activations import sigmoid_function, linear_function
from utilis.sparse_activations import (
    SparseActivationType, SparseActivationFactory, BaseSparseActivation
)


class ImprovedSparseLayer(LinearLayer):
    """Improved sparse layer with configurable activation types and initialization."""
    
    def __init__(self, name: str, n_in: int, n_out: int, 
                 activation: Callable[[np.ndarray, bool], np.ndarray] = sigmoid_function,
                 num_k_sparse: int = 10,
                 sparse_activation_type: SparseActivationType = SparseActivationType.JUMP_RELU,
                 initialization_method: str = "tied",
                 decoder_layer: Optional['LinearLayer'] = None,
                 **sparse_kwargs):
        """Initialize improved sparse layer.
        
        Args:
            name: Layer name
            n_in: Number of input neurons
            n_out: Number of output neurons
            activation: Activation function
            num_k_sparse: Number of neurons to keep active
            sparse_activation_type: Type of sparse activation to use
            initialization_method: Weight initialization method
            decoder_layer: Reference to decoder layer for tied initialization
            **sparse_kwargs: Additional arguments for sparse activation
        """
        super().__init__(name, n_in, n_out, activation)
        
        self.num_k_sparse = num_k_sparse
        self.sparse_activation_type = sparse_activation_type
        self.initialization_method = initialization_method
        self.decoder_layer = decoder_layer
        
        # Create sparse activation function
        self.sparse_activation = SparseActivationFactory.create_sparse_activation(
            sparse_activation_type, num_k_sparse, n_features=n_out, n_input=n_in, **sparse_kwargs
        )
        
        # Initialize weights and biases
        self._initialize_parameters()
        
        # Store for gradient computation
        self.sparsity_mask = None
        self.pre_activations = None
        self.sparse_activations = None
    
    def _initialize_parameters(self):
        """Initialize parameters based on the specified method."""
        if self.initialization_method == "tied" and self.decoder_layer is not None:
            # Tied initialization: encoder weights are transpose of decoder weights
            self.weights = (self.n_out / self.n_in) * self.decoder_layer.weights.T
            
            # Initialize biases to zero for tied initialization
            self.biases = np.zeros(self.n_out)
            
        elif self.initialization_method == "xavier":
            # Xavier initialization
            fan_in, fan_out = self.n_in, self.n_out
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = np.random.uniform(-limit, limit, (fan_in, fan_out))
            self.biases = np.zeros(fan_out)
            
        elif self.initialization_method == "he":
            # He initialization
            self.weights = np.random.normal(0, np.sqrt(2 / self.n_in), (self.n_in, self.n_out))
            self.biases = np.zeros(self.n_out)
            
        elif self.initialization_method == "sparse_friendly":
            # Sparse-friendly initialization
            self.weights = np.random.normal(0, 0.1, (self.n_in, self.n_out))
            
            # Initialize biases to encourage sparse activation
            target_activation_rate = self.num_k_sparse / self.n_out
            self.biases = np.random.normal(-np.log(1/target_activation_rate - 1), 0.1, self.n_out)
        
        # If no specific initialization or default case
        if not hasattr(self, 'weights') or self.weights is None:
            # Standard initialization
            self.weights = np.random.normal(0, 0.1, (self.n_in, self.n_out))
            self.biases = np.zeros(self.n_out)
    
    def initialize_data_driven_biases(self, data_sample: np.ndarray):
        """Initialize biases based on data statistics.
        
        Args:
            data_sample: Sample of input data for statistics
        """
        if self.initialization_method in ["tied", "data_driven"]:
            # Compute activations for data sample
            activations = data_sample.dot(self.weights)
            
            # Set biases to achieve target activation rate
            target_activation_rate = self.num_k_sparse / self.n_out
            
            for i in range(self.n_out):
                # Set bias so that target percentage of samples would activate
                percentile = (1 - target_activation_rate) * 100
                self.biases[i] = -np.percentile(activations[:, i], percentile)
    
    def get_output(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through improved sparse layer.
        
        Args:
            x: Input data
            
        Returns:
            Sparse layer output
        """
        # Standard linear transformation
        linear_output = x.dot(self.weights) + self.biases
        
        # Apply activation function
        activated_output = self.activation(linear_output)
        
        # Store pre-activations for loss computation
        self.pre_activations = activated_output.copy()
        
        # Apply sparse activation
        if self.sparse_activation_type == SparseActivationType.GATED_SPARSE:
            # Gated sparse needs original input
            sparse_output, mask = self.sparse_activation.forward(activated_output, x)
        else:
            sparse_output, mask = self.sparse_activation.forward(activated_output)
        
        # Store for gradient computation and loss calculation
        self.sparsity_mask = mask
        self.sparse_activations = sparse_output.copy()
        
        return sparse_output
    
    def backward_sparse(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through sparse activation.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient for sparse activation
        """
        if self.sparsity_mask is not None:
            return self.sparse_activation.backward(grad_output, self.sparsity_mask)
        else:
            return grad_output
    
    def update_sparse_parameters(self, learning_rate: float, **kwargs):
        """Update sparse activation parameters if any.
        
        Args:
            learning_rate: Learning rate for updates
            **kwargs: Additional update parameters
        """
        self.sparse_activation.update_parameters(learning_rate, **kwargs)
    
    def get_sparsity_info(self) -> Dict[str, Any]:
        """Get information about current sparsity state.
        
        Returns:
            Dictionary with sparsity information
        """
        info = {
            'target_k': self.num_k_sparse,
            'activation_type': self.sparse_activation_type,
            'initialization_method': self.initialization_method
        }
        
        if self.sparsity_mask is not None:
            active_neurons = np.sum(self.sparsity_mask, axis=1)
            info.update({
                'actual_k_mean': np.mean(active_neurons),
                'actual_k_std': np.std(active_neurons),
                'actual_k_min': np.min(active_neurons),
                'actual_k_max': np.max(active_neurons)
            })
        
        if self.sparse_activations is not None:
            info.update({
                'activation_mean': np.mean(self.sparse_activations),
                'activation_std': np.std(self.sparse_activations),
                'zero_fraction': np.mean(self.sparse_activations == 0)
            })
        
        return info
    
    def get_learnable_thresholds(self) -> Optional[np.ndarray]:
        """Get learnable thresholds if using JumpReLU.
        
        Returns:
            Threshold values or None if not applicable
        """
        if (self.sparse_activation_type == SparseActivationType.JUMP_RELU and 
            hasattr(self.sparse_activation, 'thresholds')):
            return self.sparse_activation.thresholds.copy()
        return None
    
    def set_learnable_thresholds(self, thresholds: np.ndarray):
        """Set learnable thresholds for JumpReLU.
        
        Args:
            thresholds: New threshold values
        """
        if (self.sparse_activation_type == SparseActivationType.JUMP_RELU and 
            hasattr(self.sparse_activation, 'thresholds')):
            self.sparse_activation.thresholds = thresholds.copy()
    
    def get_dead_neurons(self, threshold: float = 1e-6) -> np.ndarray:
        """Identify dead neurons based on activation statistics.
        
        Args:
            threshold: Threshold for considering a neuron dead
            
        Returns:
            Boolean array indicating dead neurons
        """
        if self.sparse_activations is not None:
            # A neuron is dead if it rarely activates
            activation_rate = np.mean(self.sparse_activations > threshold, axis=0)
            return activation_rate < threshold
        return np.zeros(self.n_out, dtype=bool)
    
    def reset_dead_neurons(self, data_sample: np.ndarray, dead_threshold: float = 1e-6):
        """Reset dead neurons by reinitializing their weights.
        
        Args:
            data_sample: Sample of input data
            dead_threshold: Threshold for considering a neuron dead
        """
        dead_neurons = self.get_dead_neurons(dead_threshold)
        
        if np.any(dead_neurons):
            # Reinitialize weights for dead neurons
            n_dead = np.sum(dead_neurons)
            self.weights[:, dead_neurons] = np.random.normal(0, 0.1, (self.n_in, n_dead))
            
            # Reinitialize biases for dead neurons
            if self.initialization_method in ["tied", "data_driven"]:
                # Use data-driven initialization for biases
                activations = data_sample.dot(self.weights[:, dead_neurons])
                target_rate = self.num_k_sparse / self.n_out
                
                for i, neuron_idx in enumerate(np.where(dead_neurons)[0]):
                    percentile = (1 - target_rate) * 100
                    self.biases[neuron_idx] = -np.percentile(activations[:, i], percentile)
            else:
                self.biases[dead_neurons] = np.random.normal(0, 0.1, n_dead)
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return (f"ImprovedSparseLayer(name='{self.name}', n_in={self.n_in}, "
                f"n_out={self.n_out}, k={self.num_k_sparse}, "
                f"activation_type={self.sparse_activation_type.value}, "
                f"init_method='{self.initialization_method}')")