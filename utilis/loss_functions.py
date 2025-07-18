"""Advanced loss functions for K-Sparse AutoEncoders.

This module provides configurable loss functions to improve sparse representation learning,
including auxiliary losses to reduce dead neurons and improve reconstruction quality.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum


class LossType(Enum):
    """Enumeration of available loss function types."""
    BASIC_MSE = "basic_mse"
    AUXK_LOSS = "auxk_loss"
    DIVERSITY_LOSS = "diversity_loss"
    COMPREHENSIVE_LOSS = "comprehensive_loss"


class BaseLossFunction:
    """Base class for loss functions."""
    
    def __init__(self, **kwargs):
        """Initialize loss function with configuration parameters."""
        pass
    
    def compute_loss(self, x: np.ndarray, reconstruction: np.ndarray, 
                    sparse_activations: np.ndarray, pre_activations: np.ndarray, 
                    k: int, **kwargs) -> float:
        """Compute loss value.
        
        Args:
            x: Original input data
            reconstruction: Reconstructed output
            sparse_activations: Sparse layer activations after top-k selection
            pre_activations: Pre-activation values before sparsity constraint
            k: Sparsity parameter
            **kwargs: Additional parameters
            
        Returns:
            Loss value
        """
        raise NotImplementedError
    
    def compute_gradients(self, x: np.ndarray, reconstruction: np.ndarray,
                         sparse_activations: np.ndarray, pre_activations: np.ndarray,
                         k: int, **kwargs) -> np.ndarray:
        """Compute gradients for backpropagation.
        
        Args:
            x: Original input data
            reconstruction: Reconstructed output
            sparse_activations: Sparse layer activations after top-k selection
            pre_activations: Pre-activation values before sparsity constraint
            k: Sparsity parameter
            **kwargs: Additional parameters
            
        Returns:
            Gradient array
        """
        # Default: use reconstruction error gradients
        return 2 * (reconstruction - x) / len(x)


class BasicMSELoss(BaseLossFunction):
    """Basic Mean Squared Error loss function."""
    
    def __init__(self, **kwargs):
        """Initialize basic MSE loss."""
        super().__init__(**kwargs)
    
    def compute_loss(self, x: np.ndarray, reconstruction: np.ndarray, 
                    sparse_activations: np.ndarray, pre_activations: np.ndarray, 
                    k: int, **kwargs) -> float:
        """Compute MSE loss."""
        return np.mean((x - reconstruction) ** 2)


class AuxKLoss(BaseLossFunction):
    """Auxiliary K loss to reduce dead neurons and improve feature utilization."""
    
    def __init__(self, mse_coeff: float = 1.0, auxk_coeff: float = 0.02, 
                 l1_coeff: float = 0.01, **kwargs):
        """Initialize AuxK loss function.
        
        Args:
            mse_coeff: Weight for reconstruction loss
            auxk_coeff: Weight for auxiliary k loss
            l1_coeff: Weight for L1 sparsity penalty
        """
        super().__init__(**kwargs)
        self.mse_coeff = mse_coeff
        self.auxk_coeff = auxk_coeff
        self.l1_coeff = l1_coeff
    
    def compute_loss(self, x: np.ndarray, reconstruction: np.ndarray, 
                    sparse_activations: np.ndarray, pre_activations: np.ndarray, 
                    k: int, **kwargs) -> float:
        """Compute AuxK loss with multiple components."""
        # Reconstruction loss
        recon_loss = np.mean((x - reconstruction) ** 2)
        
        # L1 sparsity penalty
        l1_loss = np.mean(np.abs(sparse_activations))
        
        # AuxK loss - encourages top-k features to be more active
        batch_size = pre_activations.shape[0]
        if k > 0 and k < pre_activations.shape[1]:
            # Get top-k indices for each sample
            topk_indices = np.argpartition(pre_activations, -k, axis=1)[:, -k:]
            batch_indices = np.arange(batch_size)[:, np.newaxis]
            
            # Compute difference between pre-activations and sparse activations for top-k
            topk_pre = pre_activations[batch_indices, topk_indices]
            topk_sparse = sparse_activations[batch_indices, topk_indices]
            auxk_loss = np.mean((topk_pre - topk_sparse) ** 2)
        else:
            auxk_loss = 0.0
        
        return (self.mse_coeff * recon_loss + 
                self.l1_coeff * l1_loss + 
                self.auxk_coeff * auxk_loss)


class DiversityLoss(BaseLossFunction):
    """Feature diversity loss to encourage different features to capture different aspects."""
    
    def __init__(self, mse_coeff: float = 1.0, diversity_coeff: float = 0.01, 
                 l1_coeff: float = 0.01, **kwargs):
        """Initialize diversity loss function.
        
        Args:
            mse_coeff: Weight for reconstruction loss
            diversity_coeff: Weight for feature diversity loss
            l1_coeff: Weight for L1 sparsity penalty
        """
        super().__init__(**kwargs)
        self.mse_coeff = mse_coeff
        self.diversity_coeff = diversity_coeff
        self.l1_coeff = l1_coeff
    
    def compute_loss(self, x: np.ndarray, reconstruction: np.ndarray, 
                    sparse_activations: np.ndarray, pre_activations: np.ndarray, 
                    k: int, **kwargs) -> float:
        """Compute diversity loss."""
        # Reconstruction loss
        recon_loss = np.mean((x - reconstruction) ** 2)
        
        # L1 sparsity penalty
        l1_loss = np.mean(np.abs(sparse_activations))
        
        # Feature diversity loss
        diversity_loss = self._compute_diversity_loss(sparse_activations)
        
        return (self.mse_coeff * recon_loss + 
                self.l1_coeff * l1_loss + 
                self.diversity_coeff * diversity_loss)
    
    def _compute_diversity_loss(self, activations: np.ndarray) -> float:
        """Compute feature diversity loss to encourage uncorrelated features."""
        if activations.shape[0] <= 1:
            return 0.0
        
        # Center activations
        centered = activations - np.mean(activations, axis=0)
        
        # Avoid division by zero
        std_vals = np.std(centered, axis=0)
        if np.all(std_vals <= 1e-8):
            return 0.0
        
        # Only use features with non-zero variance
        valid_features = std_vals > 1e-8
        if np.sum(valid_features) <= 1:
            return 0.0
        
        centered_valid = centered[:, valid_features]
        
        # Compute correlation matrix
        try:
            correlation_matrix = np.corrcoef(centered_valid.T)
            
            # Handle NaN values and ensure finite values
            if np.any(np.isnan(correlation_matrix)) or np.any(np.isinf(correlation_matrix)):
                return 0.0
            
            # Penalize high correlations (excluding diagonal)
            mask = np.ones_like(correlation_matrix) - np.eye(correlation_matrix.shape[0])
            diversity_loss = np.mean(correlation_matrix ** 2 * mask)
            
            return diversity_loss
        except:
            return 0.0


class ComprehensiveLoss(BaseLossFunction):
    """Comprehensive loss combining multiple objectives for optimal sparse learning."""
    
    def __init__(self, mse_coeff: float = 1.0, auxk_coeff: float = 0.02, 
                 diversity_coeff: float = 0.01, l1_coeff: float = 0.01, 
                 dead_neuron_coeff: float = 0.001, **kwargs):
        """Initialize comprehensive loss function.
        
        Args:
            mse_coeff: Weight for reconstruction loss
            auxk_coeff: Weight for auxiliary k loss
            diversity_coeff: Weight for feature diversity loss
            l1_coeff: Weight for L1 sparsity penalty
            dead_neuron_coeff: Weight for dead neuron penalty
        """
        super().__init__(**kwargs)
        self.mse_coeff = mse_coeff
        self.auxk_coeff = auxk_coeff
        self.diversity_coeff = diversity_coeff
        self.l1_coeff = l1_coeff
        self.dead_neuron_coeff = dead_neuron_coeff
    
    def compute_loss(self, x: np.ndarray, reconstruction: np.ndarray, 
                    sparse_activations: np.ndarray, pre_activations: np.ndarray, 
                    k: int, **kwargs) -> float:
        """Compute comprehensive loss with all components."""
        # Reconstruction loss
        recon_loss = np.mean((x - reconstruction) ** 2)
        
        # L1 sparsity penalty
        l1_loss = np.mean(np.abs(sparse_activations))
        
        # AuxK loss
        auxk_loss = self._compute_auxk_loss(sparse_activations, pre_activations, k)
        
        # Feature diversity loss
        diversity_loss = self._compute_diversity_loss(sparse_activations)
        
        # Dead neuron penalty
        dead_neuron_loss = self._compute_dead_neuron_penalty(pre_activations)
        
        return (self.mse_coeff * recon_loss + 
                self.l1_coeff * l1_loss + 
                self.auxk_coeff * auxk_loss + 
                self.diversity_coeff * diversity_loss + 
                self.dead_neuron_coeff * dead_neuron_loss)
    
    def _compute_auxk_loss(self, sparse_activations: np.ndarray, 
                          pre_activations: np.ndarray, k: int) -> float:
        """Compute auxiliary k loss."""
        batch_size = pre_activations.shape[0]
        if k > 0 and k < pre_activations.shape[1]:
            topk_indices = np.argpartition(pre_activations, -k, axis=1)[:, -k:]
            batch_indices = np.arange(batch_size)[:, np.newaxis]
            
            topk_pre = pre_activations[batch_indices, topk_indices]
            topk_sparse = sparse_activations[batch_indices, topk_indices]
            return np.mean((topk_pre - topk_sparse) ** 2)
        return 0.0
    
    def _compute_diversity_loss(self, activations: np.ndarray) -> float:
        """Compute feature diversity loss."""
        if activations.shape[0] <= 1:
            return 0.0
        
        # Center activations
        centered = activations - np.mean(activations, axis=0)
        
        # Avoid division by zero
        std_vals = np.std(centered, axis=0)
        if np.all(std_vals <= 1e-8):
            return 0.0
        
        # Only use features with non-zero variance
        valid_features = std_vals > 1e-8
        if np.sum(valid_features) <= 1:
            return 0.0
        
        centered_valid = centered[:, valid_features]
        
        # Compute correlation matrix
        try:
            correlation_matrix = np.corrcoef(centered_valid.T)
            
            # Handle NaN values and ensure finite values
            if np.any(np.isnan(correlation_matrix)) or np.any(np.isinf(correlation_matrix)):
                return 0.0
            
            # Penalize high correlations (excluding diagonal)
            mask = np.ones_like(correlation_matrix) - np.eye(correlation_matrix.shape[0])
            diversity_loss = np.mean(correlation_matrix ** 2 * mask)
            
            return diversity_loss
        except:
            return 0.0
    
    def _compute_dead_neuron_penalty(self, pre_activations: np.ndarray) -> float:
        """Compute penalty for dead neurons."""
        # Penalize neurons that are consistently inactive
        return np.mean(np.maximum(0, -pre_activations))


class LossFactory:
    """Factory class for creating loss functions."""
    
    @staticmethod
    def create_loss_function(loss_type: LossType, **kwargs) -> BaseLossFunction:
        """Create a loss function of the specified type.
        
        Args:
            loss_type: Type of loss function to create
            **kwargs: Configuration parameters for the loss function
            
        Returns:
            Configured loss function instance
        """
        if loss_type == LossType.BASIC_MSE:
            return BasicMSELoss(**kwargs)
        elif loss_type == LossType.AUXK_LOSS:
            return AuxKLoss(**kwargs)
        elif loss_type == LossType.DIVERSITY_LOSS:
            return DiversityLoss(**kwargs)
        elif loss_type == LossType.COMPREHENSIVE_LOSS:
            return ComprehensiveLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


# For backward compatibility
def subtract_err(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Backward compatibility function for basic MSE gradient."""
    return 2 * (actual - expected) / len(expected)