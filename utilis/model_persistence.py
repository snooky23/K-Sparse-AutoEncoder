"""Model persistence and loading system for K-Sparse AutoEncoder.

This module provides comprehensive model saving and loading functionality,
including metadata, configuration, and training history.
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import hashlib

from layers.linear_layer import LinearLayer
from layers.sparse_layer import SparseLayer
from layers.improved_sparse_layer import ImprovedSparseLayer
from nets.fcnn import FCNeuralNet
from nets.improved_fcnn import ImprovedFCNN
from utilis.config import ExperimentConfig
from utilis.loss_functions import LossType
from utilis.sparse_activations import SparseActivationType


class ModelMetadata:
    """Model metadata container."""
    
    def __init__(self, model_type: str, layers: List[Dict], config: Optional[Dict] = None):
        """Initialize model metadata.
        
        Args:
            model_type: Type of model (FCNeuralNet, ImprovedFCNN)
            layers: List of layer configurations
            config: Optional experiment configuration
        """
        self.model_type = model_type
        self.layers = layers
        self.config = config
        self.creation_time = datetime.now().isoformat()
        self.version = "1.0"
        self.checksum = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'model_type': self.model_type,
            'layers': self.layers,
            'config': self.config,
            'creation_time': self.creation_time,
            'version': self.version,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        metadata = cls(
            model_type=data['model_type'],
            layers=data['layers'],
            config=data.get('config')
        )
        metadata.creation_time = data.get('creation_time')
        metadata.version = data.get('version', "1.0")
        metadata.checksum = data.get('checksum')
        return metadata


class ModelPersistence:
    """Handles model saving and loading with comprehensive metadata."""
    
    @staticmethod
    def save_model(model: Union[FCNeuralNet, ImprovedFCNN], 
                   filepath: str,
                   config: Optional[ExperimentConfig] = None,
                   training_history: Optional[Dict] = None,
                   additional_data: Optional[Dict] = None) -> None:
        """Save model with comprehensive metadata.
        
        Args:
            model: Model to save
            filepath: Path to save model
            config: Optional experiment configuration
            training_history: Optional training history
            additional_data: Optional additional data to save
        """
        # Create output directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Extract model information
        model_data = ModelPersistence._extract_model_data(model)
        
        # Create metadata
        metadata = ModelMetadata(
            model_type=type(model).__name__,
            layers=model_data['layers_config'],
            config=config.__dict__ if config else None
        )
        
        # Prepare data for saving
        save_data = {
            'metadata': metadata.to_dict(),
            'model_data': model_data,
            'training_history': training_history,
            'additional_data': additional_data or {}
        }
        
        # Calculate checksum
        checksum = ModelPersistence._calculate_checksum(save_data)
        save_data['metadata']['checksum'] = checksum
        
        # Save to file
        if filepath.endswith('.npz'):
            # Save as numpy compressed format
            np.savez_compressed(filepath, **save_data)
        elif filepath.endswith('.pkl'):
            # Save as pickle
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            # Default to npz
            np.savez_compressed(filepath + '.npz', **save_data)
    
    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """Load model with metadata.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Dictionary containing model, metadata, and additional data
        """
        # Load data
        if filepath.endswith('.npz'):
            data = np.load(filepath, allow_pickle=True)
            load_data = {key: data[key].item() if data[key].ndim == 0 else data[key] 
                        for key in data.files}
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                load_data = pickle.load(f)
        else:
            # Try npz first
            try:
                data = np.load(filepath + '.npz', allow_pickle=True)
                load_data = {key: data[key].item() if data[key].ndim == 0 else data[key] 
                            for key in data.files}
            except:
                # Try pkl
                with open(filepath + '.pkl', 'rb') as f:
                    load_data = pickle.load(f)
        
        # Verify checksum
        metadata = ModelMetadata.from_dict(load_data['metadata'])
        expected_checksum = metadata.checksum
        
        # Temporarily remove checksum for verification
        temp_data = load_data.copy()
        temp_data['metadata'] = metadata.to_dict()
        temp_data['metadata']['checksum'] = None
        
        actual_checksum = ModelPersistence._calculate_checksum(temp_data)
        
        if expected_checksum and expected_checksum != actual_checksum:
            print(f"⚠️  Warning: Model checksum mismatch. File may be corrupted.")
        
        # Reconstruct model
        model = ModelPersistence._reconstruct_model(load_data['model_data'], metadata)
        
        return {
            'model': model,
            'metadata': metadata,
            'training_history': load_data.get('training_history'),
            'additional_data': load_data.get('additional_data', {})
        }
    
    @staticmethod
    def save_checkpoint(model: Union[FCNeuralNet, ImprovedFCNN],
                       checkpoint_dir: str,
                       epoch: int,
                       config: Optional[ExperimentConfig] = None,
                       training_history: Optional[Dict] = None) -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            checkpoint_dir: Directory to save checkpoints
            epoch: Current epoch
            config: Optional experiment configuration
            training_history: Optional training history
            
        Returns:
            Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.npz')
        
        additional_data = {
            'epoch': epoch,
            'checkpoint_type': 'training'
        }
        
        ModelPersistence.save_model(
            model, checkpoint_path, config, training_history, additional_data
        )
        
        return checkpoint_path
    
    @staticmethod
    def load_latest_checkpoint(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint from directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            Latest checkpoint data or None if no checkpoints found
        """
        if not os.path.exists(checkpoint_dir):
            return None
        
        # Find all checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.npz')]
        
        if not checkpoint_files:
            return None
        
        # Sort by epoch number and get latest
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_file = checkpoint_files[-1]
        
        checkpoint_path = os.path.join(checkpoint_dir, latest_file)
        return ModelPersistence.load_model(checkpoint_path)
    
    @staticmethod
    def export_model_summary(model: Union[FCNeuralNet, ImprovedFCNN],
                           filepath: str,
                           config: Optional[ExperimentConfig] = None) -> None:
        """Export model summary to JSON.
        
        Args:
            model: Model to summarize
            filepath: Path to save summary
            config: Optional experiment configuration
        """
        summary = {
            'model_type': type(model).__name__,
            'layers': [],
            'total_parameters': 0,
            'creation_time': datetime.now().isoformat()
        }
        
        # Add layer information
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': type(layer).__name__,
                'input_size': layer.n_in,
                'output_size': layer.n_out,
                'parameters': layer.weights.size + layer.biases.size
            }
            
            if isinstance(layer, (SparseLayer, ImprovedSparseLayer)):
                layer_info['num_k_sparse'] = layer.num_k_sparse
                if hasattr(layer, 'sparse_activation_type'):
                    layer_info['sparse_activation_type'] = layer.sparse_activation_type.value
            
            summary['layers'].append(layer_info)
            summary['total_parameters'] += layer_info['parameters']
        
        # Add configuration if provided
        if config:
            summary['config'] = config.__dict__
        
        # Save summary
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    @staticmethod
    def _extract_model_data(model: Union[FCNeuralNet, ImprovedFCNN]) -> Dict[str, Any]:
        """Extract model data for saving."""
        model_data = {
            'layers_config': [],
            'layers_weights': [],
            'layers_biases': []
        }
        
        # Extract layer information
        for layer in model.layers:
            layer_config = {
                'name': layer.name,
                'type': type(layer).__name__,
                'n_in': layer.n_in,
                'n_out': layer.n_out,
                'activation': layer.activation.__name__ if hasattr(layer, 'activation') else None
            }
            
            # Add sparse layer specific information
            if isinstance(layer, (SparseLayer, ImprovedSparseLayer)):
                layer_config['num_k_sparse'] = layer.num_k_sparse
                
                if isinstance(layer, ImprovedSparseLayer):
                    layer_config['sparse_activation_type'] = layer.sparse_activation_type.value
                    layer_config['initialization_method'] = layer.initialization_method
                    
                    # Save learnable thresholds if available
                    thresholds = layer.get_learnable_thresholds()
                    if thresholds is not None:
                        layer_config['learnable_thresholds'] = thresholds
            
            model_data['layers_config'].append(layer_config)
            model_data['layers_weights'].append(layer.weights)
            model_data['layers_biases'].append(layer.biases)
        
        # Add model-specific information
        if isinstance(model, ImprovedFCNN):
            model_data['loss_function_type'] = type(model.loss_function).__name__
            model_data['curriculum_learning'] = model.curriculum_learning
            model_data['curriculum_config'] = model.curriculum_config
            model_data['training_history'] = model.training_history
        
        return model_data
    
    @staticmethod
    def _reconstruct_model(model_data: Dict[str, Any], 
                          metadata: ModelMetadata) -> Union[FCNeuralNet, ImprovedFCNN]:
        """Reconstruct model from saved data."""
        # Import activation functions
        from utilis.activations import (
            sigmoid_function, relu_function, tanh_function, 
            linear_function, leaky_relu_function
        )
        
        activation_map = {
            'sigmoid_function': sigmoid_function,
            'relu_function': relu_function,
            'tanh_function': tanh_function,
            'linear_function': linear_function,
            'leaky_relu_function': leaky_relu_function
        }
        
        # Reconstruct layers
        layers = []
        for i, layer_config in enumerate(model_data['layers_config']):
            layer_type = layer_config['type']
            activation = activation_map.get(layer_config['activation'], sigmoid_function)
            
            if layer_type == 'LinearLayer':
                layer = LinearLayer(
                    name=layer_config['name'],
                    n_in=layer_config['n_in'],
                    n_out=layer_config['n_out'],
                    activation=activation
                )
            elif layer_type == 'SparseLayer':
                layer = SparseLayer(
                    name=layer_config['name'],
                    n_in=layer_config['n_in'],
                    n_out=layer_config['n_out'],
                    activation=activation,
                    num_k_sparse=layer_config['num_k_sparse']
                )
            elif layer_type == 'ImprovedSparseLayer':
                layer = ImprovedSparseLayer(
                    name=layer_config['name'],
                    n_in=layer_config['n_in'],
                    n_out=layer_config['n_out'],
                    activation=activation,
                    num_k_sparse=layer_config['num_k_sparse'],
                    sparse_activation_type=SparseActivationType(layer_config['sparse_activation_type']),
                    initialization_method=layer_config['initialization_method']
                )
                
                # Restore learnable thresholds if available
                if 'learnable_thresholds' in layer_config:
                    layer.set_learnable_thresholds(layer_config['learnable_thresholds'])
            
            # Load weights and biases
            layer.weights = model_data['layers_weights'][i]
            layer.biases = model_data['layers_biases'][i]
            
            layers.append(layer)
        
        # Reconstruct model
        if metadata.model_type == 'FCNeuralNet':
            model = FCNeuralNet(layers)
        elif metadata.model_type == 'ImprovedFCNN':
            # Reconstruct loss function
            loss_function = LossType.BASIC_MSE  # Default
            if 'loss_function_type' in model_data:
                loss_name = model_data['loss_function_type'].replace('Loss', '').lower()
                if loss_name == 'basicmse':
                    loss_function = LossType.BASIC_MSE
                elif loss_name == 'auxk':
                    loss_function = LossType.AUXK_LOSS
                elif loss_name == 'diversity':
                    loss_function = LossType.DIVERSITY_LOSS
                elif loss_name == 'comprehensive':
                    loss_function = LossType.COMPREHENSIVE_LOSS
            
            model = ImprovedFCNN(
                layers=layers,
                loss_function=loss_function,
                curriculum_learning=model_data.get('curriculum_learning', False),
                curriculum_config=model_data.get('curriculum_config')
            )
            
            # Restore training history
            if 'training_history' in model_data:
                model.training_history = model_data['training_history']
        
        return model
    
    @staticmethod
    def _calculate_checksum(data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity."""
        # Convert data to string for hashing
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()


class ModelRegistry:
    """Registry for managing multiple saved models."""
    
    def __init__(self, registry_dir: str = "models/"):
        """Initialize model registry.
        
        Args:
            registry_dir: Directory to store models
        """
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, 'registry.json')
        self.models = self._load_registry()
    
    def register_model(self, model_id: str, model_path: str, 
                      description: str = "", tags: List[str] = None) -> None:
        """Register a model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the model file
            description: Optional description
            tags: Optional list of tags
        """
        self.models[model_id] = {
            'path': model_path,
            'description': description,
            'tags': tags or [],
            'registered_at': datetime.now().isoformat()
        }
        self._save_registry()
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information or None if not found
        """
        return self.models.get(model_id)
    
    def load_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model data or None if not found
        """
        model_info = self.get_model(model_id)
        if model_info:
            return ModelPersistence.load_model(model_info['path'])
        return None
    
    def list_models(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered models.
        
        Args:
            tag: Optional tag filter
            
        Returns:
            List of model information
        """
        models = []
        for model_id, info in self.models.items():
            if tag is None or tag in info['tags']:
                models.append({
                    'id': model_id,
                    'path': info['path'],
                    'description': info['description'],
                    'tags': info['tags'],
                    'registered_at': info['registered_at']
                })
        return models
    
    def remove_model(self, model_id: str) -> bool:
        """Remove model from registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model was removed, False if not found
        """
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            return True
        return False
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        os.makedirs(self.registry_dir, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.models, f, indent=2)


# Convenience functions
def save_model(model: Union[FCNeuralNet, ImprovedFCNN], 
               filepath: str, **kwargs) -> None:
    """Convenience function to save model."""
    ModelPersistence.save_model(model, filepath, **kwargs)


def load_model(filepath: str) -> Dict[str, Any]:
    """Convenience function to load model."""
    return ModelPersistence.load_model(filepath)


def create_model_registry(registry_dir: str = "models/") -> ModelRegistry:
    """Create a model registry."""
    return ModelRegistry(registry_dir)