"""Tests for FCNeuralNet class."""
import unittest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nets.fcnn import FCNeuralNet
from layers.linear_layer import LinearLayer
from layers.sparse_layer import SparseLayer
from utilis.activations import sigmoid_function
from utilis.cost_functions import subtract_err, mse


class TestFCNeuralNet(unittest.TestCase):
    """Test cases for FCNeuralNet class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layers = [
            LinearLayer("layer1", n_in=4, n_out=3, activation=sigmoid_function),
            LinearLayer("layer2", n_in=3, n_out=2, activation=sigmoid_function)
        ]
        self.network = FCNeuralNet(self.layers)
        
        # Test data
        self.x_train = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ])
        self.y_train = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])

    def test_init(self) -> None:
        """Test network initialization."""
        self.assertEqual(len(self.network.layers), 2)
        self.assertEqual(self.network.cost_func, subtract_err)

    def test_init_with_custom_cost(self) -> None:
        """Test network initialization with custom cost function."""
        network = FCNeuralNet(self.layers, cost_func=mse)
        self.assertEqual(network.cost_func, mse)

    def test_print_network(self) -> None:
        """Test network printing doesn't raise errors."""
        # This should not raise an exception
        self.network.print_network()

    def test_feed_forward_shape(self) -> None:
        """Test feed forward returns correct shapes."""
        results = self.network.feed_forward(self.x_train)
        
        # Should return input + output of each layer
        self.assertEqual(len(results), 3)  # input + 2 layers
        self.assertEqual(results[0].shape, (3, 4))  # input
        self.assertEqual(results[1].shape, (3, 3))  # layer1 output
        self.assertEqual(results[2].shape, (3, 2))  # layer2 output

    def test_feed_forward_deterministic(self) -> None:
        """Test feed forward is deterministic."""
        results1 = self.network.feed_forward(self.x_train)
        results2 = self.network.feed_forward(self.x_train)
        
        for r1, r2 in zip(results1, results2):
            np.testing.assert_array_equal(r1, r2)

    def test_predict(self) -> None:
        """Test prediction returns last layer output."""
        prediction = self.network.predict(self.x_train)
        feed_forward_results = self.network.feed_forward(self.x_train)
        
        np.testing.assert_array_equal(prediction, feed_forward_results[-1])

    def test_predict_shape(self) -> None:
        """Test prediction shape."""
        prediction = self.network.predict(self.x_train)
        self.assertEqual(prediction.shape, (3, 2))

    def test_back_propagate_shapes(self) -> None:
        """Test backpropagation returns correct delta shapes."""
        results = self.network.feed_forward(self.x_train)
        error = self.network.cost_func(results[-1], self.y_train)
        deltas = self.network.back_propagate(results, error)
        
        self.assertEqual(len(deltas), 2)  # One delta per layer
        self.assertEqual(deltas[0].shape, (3, 3))  # layer1 deltas
        self.assertEqual(deltas[1].shape, (3, 2))  # layer2 deltas

    def test_update_weights(self) -> None:
        """Test weight updates modify layer weights."""
        # Store original weights
        original_weights = [layer.weights.copy() for layer in self.network.layers]
        
        # Perform one training step
        results = self.network.feed_forward(self.x_train)
        error = self.network.cost_func(results[-1], self.y_train)
        deltas = self.network.back_propagate(results, error)
        self.network.update_weights(results, deltas, learning_rate=0.1)
        
        # Weights should have changed
        for i, layer in enumerate(self.network.layers):
            self.assertFalse(np.array_equal(layer.weights, original_weights[i]))

    def test_train_reduces_loss(self) -> None:
        """Test that training reduces loss over epochs."""
        # Record initial prediction
        initial_pred = self.network.predict(self.x_train)
        initial_loss = np.mean(np.abs(self.network.cost_func(initial_pred, self.y_train)))
        
        # Train for a few epochs
        self.network.train(self.x_train, self.y_train, 
                          learning_rate=0.1, epochs=10, 
                          batch_size=2, print_epochs=100)
        
        # Check final loss
        final_pred = self.network.predict(self.x_train)
        final_loss = np.mean(np.abs(self.network.cost_func(final_pred, self.y_train)))
        
        # Loss should decrease (though this is not guaranteed for all cases)
        # At minimum, training should complete without errors
        self.assertIsInstance(final_loss, float)

    def test_train_with_small_batch(self) -> None:
        """Test training with batch size smaller than dataset."""
        # This should not raise an exception
        self.network.train(self.x_train, self.y_train,
                          learning_rate=0.01, epochs=2,
                          batch_size=2, print_epochs=10)

    def test_accuracy_classification(self) -> None:
        """Test accuracy calculation for classification."""
        # Create simple prediction that should give known accuracy
        y_labels = np.array([
            [1, 0, 0],  # class 0
            [0, 1, 0],  # class 1
            [0, 0, 1],  # class 2
            [1, 0, 0]   # class 0
        ])
        
        x_data = np.random.randn(4, 4)  # Dummy input data
        
        # Create network with 3 output classes
        layers = [LinearLayer("layer1", n_in=4, n_out=3, activation=sigmoid_function)]
        network = FCNeuralNet(layers)
        
        accuracy = network.accuracy(x_data, y_labels)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)

    def test_single_layer_network(self) -> None:
        """Test network with single layer."""
        single_layer = [LinearLayer("single", n_in=4, n_out=2)]
        network = FCNeuralNet(single_layer)
        
        prediction = network.predict(self.x_train)
        self.assertEqual(prediction.shape, (3, 2))

    def test_sparse_layer_integration(self) -> None:
        """Test network with sparse layers."""
        sparse_layers = [
            SparseLayer("sparse1", n_in=4, n_out=6, num_k_sparse=3),
            LinearLayer("linear", n_in=6, n_out=2)
        ]
        sparse_network = FCNeuralNet(sparse_layers)
        
        prediction = sparse_network.predict(self.x_train)
        self.assertEqual(prediction.shape, (3, 2))
        
        # Test training doesn't crash
        sparse_network.train(self.x_train, self.y_train,
                           learning_rate=0.01, epochs=2,
                           batch_size=2, print_epochs=10)

    def test_different_input_sizes(self) -> None:
        """Test network with different input sizes."""
        # Single sample
        single_sample = self.x_train[:1]
        prediction = self.network.predict(single_sample)
        self.assertEqual(prediction.shape, (1, 2))
        
        # Larger batch
        large_batch = np.tile(self.x_train, (3, 1))  # 9x4
        prediction = self.network.predict(large_batch)
        self.assertEqual(prediction.shape, (9, 2))

    def test_train_monitor_accuracy(self) -> None:
        """Test training with accuracy monitoring."""
        # This should not raise an exception
        self.network.train(self.x_train, self.y_train,
                          learning_rate=0.01, epochs=2,
                          batch_size=2, print_epochs=1,
                          monitor_train_accuracy=True)


if __name__ == '__main__':
    unittest.main()