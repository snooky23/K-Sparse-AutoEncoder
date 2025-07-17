"""Tests for activation functions."""
import unittest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilis.activations import sigmoid_function, relu_function, tanh_function, softmax_function


class TestActivationFunctions(unittest.TestCase):
    """Test cases for activation functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_input = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        self.batch_input = np.array([
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [0.0, 0.5, 1.0, 1.5, 2.0]
        ])

    def test_sigmoid_function_forward(self) -> None:
        """Test sigmoid forward pass."""
        output = sigmoid_function(self.test_input)
        
        # Check shape
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Check range [0, 1]
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
        
        # Check specific values
        np.testing.assert_almost_equal(sigmoid_function(np.array([[0.0]])), 0.5, decimal=5)
        
        # Check monotonicity (larger inputs -> larger outputs)
        sorted_input = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        sorted_output = sigmoid_function(sorted_input)
        self.assertTrue(np.all(np.diff(sorted_output[0]) > 0))

    def test_sigmoid_function_derivative(self) -> None:
        """Test sigmoid derivative."""
        forward_output = sigmoid_function(self.test_input)
        derivative_output = sigmoid_function(forward_output, derivative=True)
        
        # Check shape
        self.assertEqual(derivative_output.shape, forward_output.shape)
        
        # Check range [0, 0.25] (max derivative is at sigmoid(0) = 0.5)
        self.assertTrue(np.all(derivative_output >= 0))
        self.assertTrue(np.all(derivative_output <= 0.25))
        
        # Check that derivative at 0.5 is maximum
        sig_half = np.array([[0.5]])
        deriv_half = sigmoid_function(sig_half, derivative=True)
        np.testing.assert_almost_equal(deriv_half, 0.25, decimal=5)

    def test_relu_function_forward(self) -> None:
        """Test ReLU forward pass."""
        output = relu_function(self.test_input)
        
        # Check shape
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Check non-negative
        self.assertTrue(np.all(output >= 0))
        
        # Check specific behavior
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(output, expected)

    def test_relu_function_derivative(self) -> None:
        """Test ReLU derivative."""
        derivative_output = relu_function(self.test_input, derivative=True)
        
        # Check shape
        self.assertEqual(derivative_output.shape, self.test_input.shape)
        
        # Check binary values
        self.assertTrue(np.all(np.isin(derivative_output, [0.0, 1.0])))
        
        # Check specific behavior
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(derivative_output, expected)

    def test_tanh_function_forward(self) -> None:
        """Test tanh forward pass."""
        output = tanh_function(self.test_input)
        
        # Check shape
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Check range [-1, 1]
        self.assertTrue(np.all(output >= -1))
        self.assertTrue(np.all(output <= 1))
        
        # Check specific values
        np.testing.assert_almost_equal(tanh_function(np.array([[0.0]])), 0.0, decimal=5)
        
        # Check symmetry: tanh(-x) = -tanh(x)
        pos_input = np.array([[1.0]])
        neg_input = np.array([[-1.0]])
        pos_output = tanh_function(pos_input)
        neg_output = tanh_function(neg_input)
        np.testing.assert_almost_equal(pos_output, -neg_output, decimal=5)

    def test_tanh_function_derivative(self) -> None:
        """Test tanh derivative."""
        forward_output = tanh_function(self.test_input)
        derivative_output = tanh_function(forward_output, derivative=True)
        
        # Check shape
        self.assertEqual(derivative_output.shape, forward_output.shape)
        
        # Check range [0, 1] (max derivative is at tanh(0) = 0)
        self.assertTrue(np.all(derivative_output >= 0))
        self.assertTrue(np.all(derivative_output <= 1))
        
        # Check that derivative at 0 is maximum (1)
        tanh_zero = np.array([[0.0]])
        deriv_zero = tanh_function(tanh_zero, derivative=True)
        np.testing.assert_almost_equal(deriv_zero, 1.0, decimal=5)

    def test_softmax_function_single_sample(self) -> None:
        """Test softmax with single sample."""
        single_input = self.test_input
        output = softmax_function(single_input)
        
        # Check shape
        self.assertEqual(output.shape, single_input.shape)
        
        # Check probabilities sum to 1
        np.testing.assert_almost_equal(np.sum(output, axis=1), 1.0, decimal=5)
        
        # Check all probabilities are positive
        self.assertTrue(np.all(output > 0))

    def test_softmax_function_batch(self) -> None:
        """Test softmax with batch input."""
        output = softmax_function(self.batch_input)
        
        # Check shape
        self.assertEqual(output.shape, self.batch_input.shape)
        
        # Check probabilities sum to 1 for each sample
        sums = np.sum(output, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(sums.shape), decimal=5)
        
        # Check all probabilities are positive
        self.assertTrue(np.all(output > 0))

    def test_softmax_function_stability(self) -> None:
        """Test softmax numerical stability with large values."""
        large_input = np.array([[1000.0, 1001.0, 1002.0]])
        output = softmax_function(large_input)
        
        # Should not overflow or produce NaN/Inf
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))
        
        # Should still sum to 1
        np.testing.assert_almost_equal(np.sum(output), 1.0, decimal=5)

    def test_softmax_function_invariance(self) -> None:
        """Test softmax shift invariance."""
        # softmax(x + c) = softmax(x) for any constant c
        constant = 10.0
        shifted_input = self.test_input + constant
        
        original_output = softmax_function(self.test_input)
        shifted_output = softmax_function(shifted_input)
        
        np.testing.assert_array_almost_equal(original_output, shifted_output, decimal=5)

    def test_activation_functions_with_zeros(self) -> None:
        """Test activation functions with zero input."""
        zero_input = np.zeros((1, 5))
        
        # Sigmoid at 0 should be 0.5
        sigmoid_out = sigmoid_function(zero_input)
        np.testing.assert_array_almost_equal(sigmoid_out, 0.5 * np.ones_like(zero_input))
        
        # ReLU at 0 should be 0
        relu_out = relu_function(zero_input)
        np.testing.assert_array_equal(relu_out, zero_input)
        
        # Tanh at 0 should be 0
        tanh_out = tanh_function(zero_input)
        np.testing.assert_array_almost_equal(tanh_out, zero_input)
        
        # Softmax at 0 should be uniform
        softmax_out = softmax_function(zero_input)
        expected_uniform = np.full_like(zero_input, 1.0 / zero_input.shape[1])
        np.testing.assert_array_almost_equal(softmax_out, expected_uniform)

    def test_activation_functions_types(self) -> None:
        """Test that activation functions return numpy arrays."""
        for func in [sigmoid_function, relu_function, tanh_function, softmax_function]:
            if func == softmax_function:
                output = func(self.test_input)
            else:
                output = func(self.test_input)
                output_deriv = func(self.test_input, derivative=True)
                self.assertIsInstance(output_deriv, np.ndarray)
            
            self.assertIsInstance(output, np.ndarray)


if __name__ == '__main__':
    unittest.main()