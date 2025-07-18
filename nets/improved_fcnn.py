"""Enhanced fully connected neural network with advanced sparse learning capabilities.

This module provides an improved neural network that supports configurable loss functions,
curriculum learning, advanced initialization, and comprehensive sparse training features.
"""
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
import time
from layers.linear_layer import LinearLayer
from layers.improved_sparse_layer import ImprovedSparseLayer
from utilis.loss_functions import LossType, LossFactory, BaseLossFunction, BasicMSELoss
from utilis.sparse_activations import SparseActivationType


class ImprovedFCNN:
    """Enhanced fully connected neural network with sparse learning capabilities."""
    
    def __init__(self, layers: List[Union[LinearLayer, ImprovedSparseLayer]], 
                 loss_function: Union[LossType, BaseLossFunction] = LossType.BASIC_MSE,
                 loss_config: Optional[Dict[str, Any]] = None,
                 curriculum_learning: bool = False,
                 curriculum_config: Optional[Dict[str, Any]] = None,
                 dead_neuron_detection: bool = True,
                 dead_neuron_threshold: float = 1e-6):
        """Initialize enhanced neural network.
        
        Args:
            layers: List of network layers
            loss_function: Loss function type or instance
            loss_config: Configuration for loss function
            curriculum_learning: Whether to use curriculum learning
            curriculum_config: Configuration for curriculum learning
            dead_neuron_detection: Whether to detect and reset dead neurons
            dead_neuron_threshold: Threshold for dead neuron detection
        """
        self.layers = layers
        self.dead_neuron_detection = dead_neuron_detection
        self.dead_neuron_threshold = dead_neuron_threshold
        
        # Initialize loss function
        if isinstance(loss_function, LossType):
            loss_config = loss_config or {}
            self.loss_function = LossFactory.create_loss_function(loss_function, **loss_config)
        else:
            self.loss_function = loss_function
        
        # Initialize curriculum learning
        self.curriculum_learning = curriculum_learning
        if curriculum_learning:
            self.curriculum_config = curriculum_config or {
                'initial_k_ratio': 0.8,
                'final_k_ratio': 1.0,
                'curriculum_epochs': 100
            }
        else:
            self.curriculum_config = None
        
        # Initialize tied weights if applicable
        self._initialize_tied_weights()
        
        # Training history
        self.training_history = {
            'loss': [],
            'learning_rate': [],
            'sparsity_info': [],
            'dead_neurons': []
        }
    
    def _initialize_tied_weights(self):
        """Initialize tied weights between encoder and decoder if applicable."""
        # Find encoder-decoder pairs
        sparse_layers = [layer for layer in self.layers if isinstance(layer, ImprovedSparseLayer)]
        decoder_layers = [layer for layer in self.layers if isinstance(layer, LinearLayer) and not isinstance(layer, ImprovedSparseLayer)]
        
        # For autoencoder architecture, tie weights
        if len(sparse_layers) == 1 and len(decoder_layers) == 1:
            encoder = sparse_layers[0]
            decoder = decoder_layers[0]
            
            if encoder.initialization_method == "tied":
                # Set decoder as reference for encoder
                encoder.decoder_layer = decoder
                encoder._initialize_parameters()
    
    def _get_current_k_values(self, epoch: int) -> Dict[str, int]:
        """Get current k values for curriculum learning.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary mapping layer names to k values
        """
        k_values = {}
        
        for layer in self.layers:
            if isinstance(layer, ImprovedSparseLayer):
                if self.curriculum_learning and epoch < self.curriculum_config['curriculum_epochs']:
                    # Progressive sparsity
                    initial_ratio = self.curriculum_config['initial_k_ratio']
                    final_ratio = self.curriculum_config['final_k_ratio']
                    progress = epoch / self.curriculum_config['curriculum_epochs']
                    
                    current_ratio = initial_ratio + (final_ratio - initial_ratio) * progress
                    current_k = max(1, int(layer.num_k_sparse * current_ratio))
                else:
                    current_k = layer.num_k_sparse
                
                k_values[layer.name] = current_k
        
        return k_values
    
    def _apply_curriculum_k_values(self, k_values: Dict[str, int]):
        """Apply curriculum k values to sparse layers.
        
        Args:
            k_values: Dictionary mapping layer names to k values
        """
        for layer in self.layers:
            if isinstance(layer, ImprovedSparseLayer) and layer.name in k_values:
                # Temporarily modify the k value
                layer.sparse_activation.num_k_sparse = k_values[layer.name]
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions with the network.
        
        Args:
            x: Input data
            
        Returns:
            Network predictions
        """
        output = x
        for layer in self.layers:
            output = layer.get_output(output)
        return output
    
    def _forward_pass(self, x: np.ndarray) -> List[np.ndarray]:
        """Forward pass through the network, storing intermediate results.
        
        Args:
            x: Input data
            
        Returns:
            List of layer outputs
        """
        results = [x]
        output = x
        
        for layer in self.layers:
            output = layer.get_output(output)
            results.append(output)
        
        return results
    
    def _backward_pass(self, results: List[np.ndarray], 
                      target: np.ndarray) -> List[np.ndarray]:
        """Enhanced backward pass with sparse gradient routing.
        
        Args:
            results: Forward pass results
            target: Target values
            
        Returns:
            List of gradients for each layer
        """
        # Initialize gradient for output layer
        output_gradient = self.loss_function.compute_gradients(
            target, results[-1], 
            sparse_activations=results[-1] if len(self.layers) > 0 else None,
            pre_activations=results[-1] if len(self.layers) > 0 else None,
            k=0
        )
        
        gradients = []
        current_gradient = output_gradient
        
        # Backward pass through layers
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layer_input = results[i]
            layer_output = results[i + 1]
            
            # Compute activation gradients
            if hasattr(layer, 'activation'):
                activation_grad = layer.activation(layer_input.dot(layer.weights) + layer.biases, derivative=True)
                delta = current_gradient * activation_grad
            else:
                delta = current_gradient
            
            # Handle sparse layers
            if isinstance(layer, ImprovedSparseLayer):
                # Apply sparse gradient routing
                delta = layer.backward_sparse(delta)
            
            gradients.insert(0, delta)
            
            # Compute gradient for next layer
            if i > 0:
                current_gradient = delta.dot(layer.weights.T)
        
        return gradients
    
    def _update_weights(self, results: List[np.ndarray], 
                       gradients: List[np.ndarray], 
                       learning_rate: float):
        """Update network weights using gradients.
        
        Args:
            results: Forward pass results
            gradients: Backward pass gradients
            learning_rate: Learning rate
        """
        for i, layer in enumerate(self.layers):
            layer_input = results[i]
            gradient = gradients[i]
            
            # Update weights and biases
            layer.weights -= learning_rate * layer_input.T.dot(gradient)
            layer.biases -= learning_rate * np.sum(gradient, axis=0)
            
            # Update sparse parameters if applicable
            if isinstance(layer, ImprovedSparseLayer):
                layer.update_sparse_parameters(learning_rate)
    
    def _initialize_data_driven_parameters(self, x: np.ndarray):
        """Initialize data-driven parameters for sparse layers.
        
        Args:
            x: Sample input data
        """
        for layer in self.layers:
            if isinstance(layer, ImprovedSparseLayer):
                layer.initialize_data_driven_biases(x)
    
    def _detect_and_reset_dead_neurons(self, x: np.ndarray, epoch: int):
        """Detect and reset dead neurons in sparse layers.
        
        Args:
            x: Sample input data
            epoch: Current epoch
        """
        if not self.dead_neuron_detection:
            return
        
        dead_counts = {}
        
        for layer in self.layers:
            if isinstance(layer, ImprovedSparseLayer):
                dead_neurons = layer.get_dead_neurons(self.dead_neuron_threshold)
                dead_count = np.sum(dead_neurons)
                
                if dead_count > 0:
                    layer.reset_dead_neurons(x, self.dead_neuron_threshold)
                    dead_counts[layer.name] = dead_count
        
        if dead_counts:
            self.training_history['dead_neurons'].append({
                'epoch': epoch,
                'dead_counts': dead_counts
            })
    
    def _collect_sparsity_info(self) -> Dict[str, Any]:
        """Collect sparsity information from all sparse layers.
        
        Returns:
            Dictionary with sparsity information
        """
        sparsity_info = {}
        
        for layer in self.layers:
            if isinstance(layer, ImprovedSparseLayer):
                sparsity_info[layer.name] = layer.get_sparsity_info()
        
        return sparsity_info
    
    def train(self, x: np.ndarray, y: np.ndarray, 
              epochs: int = 1000, 
              learning_rate: float = 0.01,
              batch_size: Optional[int] = None,
              validation_split: float = 0.0,
              early_stopping_patience: int = 10,
              print_epochs: int = 100,
              collect_sparsity_info: bool = True,
              **kwargs) -> Dict[str, List]:
        """Train the network with enhanced features.
        
        Args:
            x: Training input data
            y: Training target data
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size (None for full batch)
            validation_split: Fraction of data for validation
            early_stopping_patience: Early stopping patience
            print_epochs: Print progress every N epochs
            collect_sparsity_info: Whether to collect sparsity statistics
            **kwargs: Additional training parameters
            
        Returns:
            Training history dictionary
        """
        print("Enhanced training started")
        start_time = time.time()
        
        # Initialize data-driven parameters
        self._initialize_data_driven_parameters(x)
        
        # Split data if validation is requested
        if validation_split > 0:
            n_val = int(len(x) * validation_split)
            val_x, val_y = x[:n_val], y[:n_val]
            train_x, train_y = x[n_val:], y[n_val:]
        else:
            train_x, train_y = x, y
            val_x, val_y = None, None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Get current k values for curriculum learning
            k_values = self._get_current_k_values(epoch)
            self._apply_curriculum_k_values(k_values)
            
            # Training step
            if batch_size is None:
                # Full batch training
                results = self._forward_pass(train_x)
                loss = self._compute_loss(train_x, results, k_values)
                gradients = self._backward_pass(results, train_y)
                self._update_weights(results, gradients, learning_rate)
            else:
                # Mini-batch training
                n_samples = len(train_x)
                indices = np.random.permutation(n_samples)
                epoch_loss = 0
                
                for i in range(0, n_samples, batch_size):
                    batch_idx = indices[i:i + batch_size]
                    batch_x = train_x[batch_idx]
                    batch_y = train_y[batch_idx]
                    
                    results = self._forward_pass(batch_x)
                    batch_loss = self._compute_loss(batch_x, results, k_values)
                    gradients = self._backward_pass(results, batch_y)
                    self._update_weights(results, gradients, learning_rate)
                    
                    epoch_loss += batch_loss
                
                loss = epoch_loss / (n_samples // batch_size + 1)
            
            # Store training history
            self.training_history['loss'].append(loss)
            self.training_history['learning_rate'].append(learning_rate)
            
            # Collect sparsity information
            if collect_sparsity_info:
                sparsity_info = self._collect_sparsity_info()
                self.training_history['sparsity_info'].append(sparsity_info)
            
            # Validation and early stopping
            if val_x is not None:
                val_results = self._forward_pass(val_x)
                val_loss = self._compute_loss(val_x, val_results, k_values)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Dead neuron detection and reset
            if (epoch + 1) % 50 == 0:  # Check every 50 epochs
                self._detect_and_reset_dead_neurons(train_x, epoch + 1)
            
            # Print progress
            if (epoch + 1) % print_epochs == 0:
                elapsed = time.time() - start_time
                print(f"epochs: {epoch + 1}, loss: {loss:.4f}, lr: {learning_rate:.6f}")
                
                # Print sparsity information
                if collect_sparsity_info and sparsity_info:
                    for layer_name, info in sparsity_info.items():
                        if 'actual_k_mean' in info:
                            print(f"  {layer_name}: k={info['actual_k_mean']:.1f}±{info['actual_k_std']:.1f}")
        
        elapsed_time = time.time() - start_time
        print(f"Enhanced training complete, elapsed time: {elapsed_time:.0f}s")
        
        return self.training_history
    
    def _compute_loss(self, x: np.ndarray, results: List[np.ndarray], 
                     k_values: Dict[str, int]) -> float:
        """Compute loss using the configured loss function.
        
        Args:
            x: Input data
            results: Forward pass results
            k_values: Current k values for layers
            
        Returns:
            Loss value
        """
        reconstruction = results[-1]
        
        # Get sparse activations and pre-activations from sparse layers
        sparse_activations = reconstruction
        pre_activations = reconstruction
        current_k = 0
        
        for layer in self.layers:
            if isinstance(layer, ImprovedSparseLayer):
                if hasattr(layer, 'sparse_activations') and layer.sparse_activations is not None:
                    sparse_activations = layer.sparse_activations
                if hasattr(layer, 'pre_activations') and layer.pre_activations is not None:
                    pre_activations = layer.pre_activations
                current_k = k_values.get(layer.name, layer.num_k_sparse)
                break
        
        return self.loss_function.compute_loss(
            x, reconstruction, sparse_activations, pre_activations, current_k
        )
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get detailed information about all layers.
        
        Returns:
            Dictionary with layer information
        """
        info = {}
        
        for layer in self.layers:
            layer_info = {
                'type': type(layer).__name__,
                'input_size': layer.n_in,
                'output_size': layer.n_out,
                'activation': layer.activation.__name__ if hasattr(layer, 'activation') else None
            }
            
            if isinstance(layer, ImprovedSparseLayer):
                layer_info.update({
                    'sparse_type': layer.sparse_activation_type.value,
                    'target_k': layer.num_k_sparse,
                    'initialization': layer.initialization_method
                })
                
                sparsity_info = layer.get_sparsity_info()
                layer_info.update(sparsity_info)
            
            info[layer.name] = layer_info
        
        return info
    
    def save_model(self, filepath: str):
        """Save model parameters to file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'layers': [],
            'loss_function_type': type(self.loss_function).__name__,
            'curriculum_config': self.curriculum_config,
            'training_history': self.training_history
        }
        
        for layer in self.layers:
            layer_data = {
                'name': layer.name,
                'type': type(layer).__name__,
                'weights': layer.weights,
                'biases': layer.biases,
                'n_in': layer.n_in,
                'n_out': layer.n_out
            }
            
            if isinstance(layer, ImprovedSparseLayer):
                layer_data.update({
                    'num_k_sparse': layer.num_k_sparse,
                    'sparse_activation_type': layer.sparse_activation_type.value,
                    'initialization_method': layer.initialization_method,
                    'thresholds': layer.get_learnable_thresholds()
                })
            
            model_data['layers'].append(layer_data)
        
        np.savez(filepath, **model_data)
    
    def load_model(self, filepath: str):
        """Load model parameters from file.
        
        Args:
            filepath: Path to load the model from
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Load training history
        if 'training_history' in data:
            self.training_history = data['training_history'].item()
        
        # Load layer parameters
        for i, layer_data in enumerate(data['layers']):
            layer = self.layers[i]
            layer.weights = layer_data['weights']
            layer.biases = layer_data['biases']
            
            if isinstance(layer, ImprovedSparseLayer) and 'thresholds' in layer_data:
                thresholds = layer_data['thresholds']
                if thresholds is not None:
                    layer.set_learnable_thresholds(thresholds)
    
    def __repr__(self) -> str:
        """String representation of the network."""
        layer_info = [f"{layer.name}({layer.n_in}->{layer.n_out})" for layer in self.layers]
        return f"ImprovedFCNN(layers=[{', '.join(layer_info)}], loss={type(self.loss_function).__name__})"