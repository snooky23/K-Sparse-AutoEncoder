"""Tests for cost functions."""
import unittest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilis.cost_functions import subtract_err, mse, cross_entropy_cost


class TestCostFunctions(unittest.TestCase):
    """Test cases for cost functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.outputs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1]
        ])
        self.targets = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])

    def test_subtract_err_basic(self) -> None:
        """Test basic subtract_err functionality."""
        result = subtract_err(self.outputs, self.targets)
        
        # Check shape
        self.assertEqual(result.shape, self.outputs.shape)
        
        # Check calculation
        expected = self.outputs - self.targets
        np.testing.assert_array_equal(result, expected)

    def test_subtract_err_perfect_match(self) -> None:
        """Test subtract_err with perfect predictions."""
        perfect_outputs = self.targets.copy()
        result = subtract_err(perfect_outputs, self.targets)
        
        # Should be all zeros
        np.testing.assert_array_equal(result, np.zeros_like(self.targets))

    def test_subtract_err_types(self) -> None:
        """Test subtract_err returns numpy array."""
        result = subtract_err(self.outputs, self.targets)
        self.assertIsInstance(result, np.ndarray)

    def test_mse_basic(self) -> None:
        """Test basic MSE functionality."""
        result = mse(self.outputs, self.targets)
        
        # Check shape
        self.assertEqual(result.shape, self.outputs.shape)
        
        # Check calculation
        expected = np.power(self.outputs - self.targets, 2)
        np.testing.assert_array_equal(result, expected)

    def test_mse_non_negative(self) -> None:
        """Test MSE is always non-negative."""
        result = mse(self.outputs, self.targets)
        self.assertTrue(np.all(result >= 0))

    def test_mse_perfect_match(self) -> None:
        """Test MSE with perfect predictions."""
        perfect_outputs = self.targets.copy()
        result = mse(perfect_outputs, self.targets)
        
        # Should be all zeros
        np.testing.assert_array_equal(result, np.zeros_like(self.targets))

    def test_mse_types(self) -> None:
        """Test MSE returns numpy array."""
        result = mse(self.outputs, self.targets)
        self.assertIsInstance(result, np.ndarray)

    def test_cross_entropy_basic(self) -> None:
        """Test basic cross-entropy functionality."""
        result = cross_entropy_cost(self.outputs, self.targets)
        
        # Should return a scalar
        self.assertIsInstance(result, (float, np.floating))

    def test_cross_entropy_perfect_predictions(self) -> None:
        """Test cross-entropy with perfect predictions."""
        # Perfect predictions should give low cost
        perfect_outputs = self.targets.copy()
        result = cross_entropy_cost(perfect_outputs, self.targets)
        
        # Should be close to zero (but not exactly due to clipping)
        self.assertLess(result, 0.1)

    def test_cross_entropy_worst_predictions(self) -> None:
        """Test cross-entropy with worst predictions."""
        # Worst predictions (opposite of targets)
        worst_outputs = 1.0 - self.targets
        result = cross_entropy_cost(worst_outputs, self.targets)
        
        # Should be high cost
        self.assertGreater(result, 1.0)

    def test_cross_entropy_stability(self) -> None:
        """Test cross-entropy numerical stability."""
        # Test with extreme values that could cause log(0)
        extreme_outputs = np.array([
            [1.0, 0.0],  # Exactly 1 and 0
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        
        # Should not produce NaN or Inf
        result = cross_entropy_cost(extreme_outputs, self.targets)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))

    def test_cross_entropy_with_probabilities(self) -> None:
        """Test cross-entropy with proper probability distributions."""
        # Outputs that sum to 1 (like softmax outputs)
        prob_outputs = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.95, 0.05]
        ])
        
        result = cross_entropy_cost(prob_outputs, self.targets)
        self.assertIsInstance(result, (float, np.floating))
        self.assertGreater(result, 0)

    def test_cost_functions_with_single_sample(self) -> None:
        """Test cost functions with single sample."""
        single_output = self.outputs[:1]
        single_target = self.targets[:1]
        
        # subtract_err
        result1 = subtract_err(single_output, single_target)
        self.assertEqual(result1.shape, (1, 2))
        
        # mse
        result2 = mse(single_output, single_target)
        self.assertEqual(result2.shape, (1, 2))
        
        # cross_entropy
        result3 = cross_entropy_cost(single_output, single_target)
        self.assertIsInstance(result3, (float, np.floating))

    def test_cost_functions_with_different_shapes(self) -> None:
        """Test cost functions with different output shapes."""
        # Test with 3-class classification
        multi_outputs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4]
        ])
        multi_targets = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # subtract_err
        result1 = subtract_err(multi_outputs, multi_targets)
        self.assertEqual(result1.shape, (3, 3))
        
        # mse
        result2 = mse(multi_outputs, multi_targets)
        self.assertEqual(result2.shape, (3, 3))
        
        # cross_entropy
        result3 = cross_entropy_cost(multi_outputs, multi_targets)
        self.assertIsInstance(result3, (float, np.floating))

    def test_cost_functions_mathematical_properties(self) -> None:
        """Test mathematical properties of cost functions."""
        # Test that MSE is always >= subtract_err^2 when positive
        sub_result = subtract_err(self.outputs, self.targets)
        mse_result = mse(self.outputs, self.targets)
        
        np.testing.assert_array_almost_equal(mse_result, np.power(sub_result, 2))

    def test_cost_functions_edge_cases(self) -> None:
        """Test cost functions with edge cases."""
        # Zero outputs and targets
        zero_outputs = np.zeros((2, 2))
        zero_targets = np.zeros((2, 2))
        
        result1 = subtract_err(zero_outputs, zero_targets)
        np.testing.assert_array_equal(result1, np.zeros((2, 2)))
        
        result2 = mse(zero_outputs, zero_targets)
        np.testing.assert_array_equal(result2, np.zeros((2, 2)))
        
        # Cross-entropy with zeros needs special handling due to clipping
        result3 = cross_entropy_cost(zero_outputs, zero_targets)
        self.assertIsInstance(result3, (float, np.floating))


if __name__ == '__main__':
    unittest.main()