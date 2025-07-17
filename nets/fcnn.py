"""Fully connected neural network implementation.

This module provides a complete neural network with training, prediction,
and evaluation capabilities for both classification and autoencoder tasks.
"""
from typing import List, Callable
import numpy as np
import time
from utilis.cost_functions import subtract_err


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
              monitor_train_accuracy: bool = False) -> None:
        """Train the neural network.
        
        Args:
            x: Training input data (n_samples, n_features)
            y: Training target data (n_samples, n_outputs)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            print_epochs: Print progress every N epochs
            monitor_train_accuracy: Whether to compute training accuracy
        """
        print("training start")
        start_time = time.time()

        for k in range(epochs):
            rand_indices = np.random.randint(x.shape[0], size=batch_size)
            batch_x = x[rand_indices]
            batch_y = y[rand_indices]

            results = self.feed_forward(batch_x)

            error = self.cost_func(results[-1], batch_y)

            if (k+1) % print_epochs == 0:
                loss = np.mean(np.abs(error))
                msg = "epochs: {0}, loss: {1:.4f}".format((k+1), loss)
                if monitor_train_accuracy:
                    accuracy = self.accuracy(x, y)
                    msg += ", accuracy: {0:.2f}%".format(accuracy)
                print(msg)
            deltas = self.back_propagate(results, error)
            self.update_weights(results, deltas, learning_rate)

        end_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
        print("training complete, elapsed time:", elapsed_time)

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
        # 1. Multiply its output delta and input activation to get the gradient of the weight.
        # 2. Subtract a ratio (percentage) of the gradient from the weight.
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_result = results[i]
            delta = deltas[i]
            layer.weights -= learning_rate * layer_result.T.dot(delta)
            # layer.biases += delta

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
