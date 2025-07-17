"""Tests for LinearLayer class."""
import unittest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.linear_layer import LinearLayer
from utilis.activations import sigmoid_function, relu_function, tanh_function


class TestLinearLayer(unittest.TestCase):
    """Test cases for LinearLayer class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layer = LinearLayer("test_layer", n_in=3, n_out=2)
        self.test_input = np.array([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0]])

    def test_init(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.name, "test_layer")
        self.assertEqual(self.layer.weights.shape, (3, 2))
        self.assertEqual(self.layer.biases.shape, (2,))
        self.assertEqual(self.layer.activation, sigmoid_function)
        self.assertTrue(isinstance(self.layer.result, np.ndarray))

    def test_weights_initialization(self) -> None:
        """Test that weights are initialized in correct range."""
        # Weights should be in range [-1, 1]
        self.assertTrue(np.all(self.layer.weights >= -1))
        self.assertTrue(np.all(self.layer.weights <= 1))

    def test_biases_initialization(self) -> None:
        """Test that biases are initialized to zero."""
        np.testing.assert_array_equal(self.layer.biases, np.zeros(2))

    def test_get_output_shape(self) -> None:
        """Test output shape is correct."""
        output = self.layer.get_output(self.test_input)
        self.assertEqual(output.shape, (2, 2))

    def test_get_output_caching(self) -> None:
        """Test that output is cached in result attribute."""
        output = self.layer.get_output(self.test_input)
        np.testing.assert_array_equal(self.layer.result, output)

    def test_different_activations(self) -> None:
        """Test layer with different activation functions."""
        relu_layer = LinearLayer("relu_layer", 3, 2, relu_function)
        tanh_layer = LinearLayer("tanh_layer", 3, 2, tanh_function)
        
        relu_output = relu_layer.get_output(self.test_input)
        tanh_output = tanh_layer.get_output(self.test_input)
        
        self.assertEqual(relu_output.shape, (2, 2))
        self.assertEqual(tanh_output.shape, (2, 2))
        
        # ReLU should have non-negative outputs for positive inputs
        self.assertTrue(np.all(relu_output >= 0))
        
        # Tanh should have outputs in range [-1, 1]
        self.assertTrue(np.all(tanh_output >= -1))
        self.assertTrue(np.all(tanh_output <= 1))

    def test_single_sample(self) -> None:
        """Test with single sample input."""
        single_input = np.array([[1.0, 2.0, 3.0]])
        output = self.layer.get_output(single_input)
        self.assertEqual(output.shape, (1, 2))

    def test_deterministic_output(self) -> None:
        """Test that same input produces same output."""
        output1 = self.layer.get_output(self.test_input)
        output2 = self.layer.get_output(self.test_input)
        np.testing.assert_array_equal(output1, output2)

    def test_batch_processing(self) -> None:
        """Test processing multiple batches."""
        batch1 = self.test_input[:1]  # First sample
        batch2 = self.test_input[1:]  # Second sample
        
        output1 = self.layer.get_output(batch1)
        output2 = self.layer.get_output(batch2)
        full_output = self.layer.get_output(self.test_input)
        
        # First row of full output should match batch1 output
        np.testing.assert_array_almost_equal(full_output[:1], output1)
        # Second row should match batch2 output
        np.testing.assert_array_almost_equal(full_output[1:], output2)


if __name__ == '__main__':
    unittest.main()