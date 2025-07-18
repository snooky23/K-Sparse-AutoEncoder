"""Fully connected neural network implementation.

This module provides a complete neural network with training, prediction,
and evaluation capabilities for both classification and autoencoder tasks.
"""
from typing import List, Callable, Optional, Tuple
import numpy as np
import time
from utilis.cost_functions import subtract_err
from utilis.regularization import (
    l1_regularization, l2_regularization, EarlyStopping, 
    LearningRateScheduler, gradient_clipping
)


class FCNeuralNet:
    """Fully connected neural network.
    
    Supports multiple layers, different cost functions, and provides
    training with mini-batch gradient descent.
    
    Attributes:
        layers: List of network layers
        cost_func: Cost function for training
    """
    def __init__(self, layers: List, cost_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = subtract_err) -> None:
        """Initialize neural network.
        
        Args:
            layers: List of layer objects
            cost_func: Cost function for computing loss
        """
        self.layers = layers
        self.cost_func = cost_func

    def print_network(self) -> None:
        """Print network architecture information."""
        print("network:")
        for layer in self.layers:
            print("layer - %s: weights: %s" % (layer.name, layer.weights.shape))

    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 10000,
              batch_size: int = 256, print_epochs: int = 1000,
              monitor_train_accuracy: bool = False, validation_split: float = 0.0,
              l1_reg: float = 0.0, l2_reg: float = 0.0, early_stopping_patience: int = 0,
              lr_schedule: Optional[str] = None, gradient_clip_norm: float = 0.0) -> dict:
        """Train the neural network with advanced features.
        
        Args:
            x: Training input data (n_samples, n_features)
            y: Training target data (n_samples, n_outputs)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            print_epochs: Print progress every N epochs
            monitor_train_accuracy: Whether to compute training accuracy
            validation_split: Fraction of data to use for validation
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            early_stopping_patience: Early stopping patience (0 = disabled)
            lr_schedule: Learning rate schedule ("step", "exponential", "cosine")
            gradient_clip_norm: Gradient clipping norm (0 = disabled)
            
        Returns:
            Dictionary containing training history
        """
        print("training start")
        start_time = time.time()
        
        # Split data for validation if requested
        if validation_split > 0.0:
            split_idx = int(x.shape[0] * (1 - validation_split))
            indices = np.random.permutation(x.shape[0])
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None
        
        n_samples = x_train.shape[0]
        effective_batch_size = min(batch_size, n_samples)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Initialize early stopping
        early_stopping = None
        if early_stopping_patience > 0:
            early_stopping = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)
        
        # Initialize learning rate scheduler
        initial_lr = learning_rate
        
        for k in range(epochs):
            # Update learning rate based on schedule
            if lr_schedule == "step":
                current_lr = LearningRateScheduler.step_decay(initial_lr, epoch=k)
            elif lr_schedule == "exponential":
                current_lr = LearningRateScheduler.exponential_decay(initial_lr, epoch=k)
            elif lr_schedule == "cosine":
                current_lr = LearningRateScheduler.cosine_annealing(initial_lr, epochs, epoch=k)
            else:
                current_lr = learning_rate
            
            # Sample batch
            if effective_batch_size == n_samples:
                batch_x, batch_y = x_train, y_train
            else:
                rand_indices = np.random.choice(n_samples, size=effective_batch_size, replace=False)
                batch_x = x_train[rand_indices]
                batch_y = y_train[rand_indices]

            # Forward pass
            results = self.feed_forward(batch_x)
            error = self.cost_func(results[-1], batch_y)
            
            # Compute base loss
            base_loss = np.mean(np.abs(error))
            
            # Add regularization penalties
            weights = [layer.weights for layer in self.layers]
            reg_loss = 0.0
            if l1_reg > 0:
                reg_loss += l1_regularization(weights, l1_reg)
            if l2_reg > 0:
                reg_loss += l2_regularization(weights, l2_reg)
            
            total_loss = base_loss + reg_loss

            # Backward pass
            deltas = self.back_propagate(results, error)
            
            # Apply gradient clipping if specified
            if gradient_clip_norm > 0:
                # Convert deltas to gradients for clipping
                gradients = []
                for i, delta in enumerate(deltas):
                    gradients.append(results[i].T.dot(delta))
                
                gradients = gradient_clipping(gradients, gradient_clip_norm)
                
                # Update weights with clipped gradients
                for i, layer in enumerate(self.layers):
                    layer.weights -= current_lr * gradients[i]
                    bias_gradient = np.mean(deltas[i], axis=0)
                    layer.biases -= current_lr * bias_gradient
            else:
                # Standard weight update
                self.update_weights(results, deltas, current_lr)
            
            # Record training metrics
            history['train_loss'].append(total_loss)
            history['learning_rates'].append(current_lr)
            
            # Compute validation metrics if validation set exists
            val_loss = None
            if x_val is not None:
                val_pred = self.predict(x_val)
                val_error = self.cost_func(val_pred, y_val)
                val_loss = np.mean(np.abs(val_error))
                history['val_loss'].append(val_loss)
            
            # Print progress
            if (k+1) % print_epochs == 0:
                msg = "epochs: {0}, loss: {1:.4f}, lr: {2:.6f}".format((k+1), total_loss, current_lr)
                if monitor_train_accuracy:
                    accuracy = self.accuracy(x_train, y_train)
                    history['train_accuracy'].append(accuracy)
                    msg += ", train_acc: {0:.2f}%".format(accuracy)
                
                if x_val is not None:
                    val_accuracy = self.accuracy(x_val, y_val)
                    history['val_accuracy'].append(val_accuracy)
                    msg += ", val_loss: {0:.4f}, val_acc: {1:.2f}%".format(val_loss, val_accuracy)
                
                print(msg)
            
            # Early stopping check
            if early_stopping is not None and x_val is not None:
                current_weights = [layer.weights for layer in self.layers]
                if early_stopping(val_loss, current_weights):
                    print(f"Early stopping at epoch {k+1}")
                    # Restore best weights
                    best_weights = early_stopping.get_best_weights()
                    if best_weights:
                        for i, layer in enumerate(self.layers):
                            layer.weights = best_weights[i]
                    break

        end_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
        print("training complete, elapsed time:", elapsed_time)
        
        return history

    def feed_forward(self, x: np.ndarray) -> List[np.ndarray]:
        """Forward pass through all layers.
        
        Args:
            x: Input data
            
        Returns:
            List of layer outputs including input
        """
        results = [x]
        for i in range(len(self.layers)):
            output_result = self.layers[i].get_output(results[i])
            results.append(output_result)
        return results

    def back_propagate(self, results: List[np.ndarray], error: np.ndarray) -> List[np.ndarray]:
        """Backpropagation to compute gradients.
        
        Args:
            results: Forward pass results from feed_forward
            error: Output error from cost function
            
        Returns:
            List of delta values for each layer
        """
        last_layer = self.layers[-1]
        deltas = [error * last_layer.activation(results[-1], derivative=True)]

        # we need to begin at the second to last layer - (a layer before the output layer)
        for i in range(len(results) - 2, 0, -1):
            layer = self.layers[i]
            delta = deltas[-1].dot(layer.weights.T) * layer.activation(results[i], derivative=True)
            deltas.append(delta)

        deltas.reverse()
        return deltas

    def update_weights(self, results: List[np.ndarray], deltas: List[np.ndarray], learning_rate: float) -> None:
        """Update network weights using computed gradients.
        
        Args:
            results: Forward pass results
            deltas: Backpropagation deltas
            learning_rate: Learning rate for updates
        """
        # Optimized weight updates with better memory usage
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_result = results[i]
            delta = deltas[i]
            
            # More efficient gradient computation
            gradient = layer_result.T.dot(delta)
            layer.weights -= learning_rate * gradient
            
            # Update biases (uncommented for completeness)
            bias_gradient = np.mean(delta, axis=0)
            layer.biases -= learning_rate * bias_gradient

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions on input data.
        
        Args:
            x: Input data for prediction
            
        Returns:
            Network predictions
        """
        return self.feed_forward(x)[-1]

    def accuracy(self, x_data: np.ndarray, y_labels: np.ndarray) -> float:
        """Compute classification accuracy.
        
        Args:
            x_data: Input data
            y_labels: True labels (one-hot encoded)
            
        Returns:
            Accuracy percentage
        """
        predictions = np.argmax(self.predict(x_data), axis=1)
        labels = np.argmax(y_labels, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy * 100
