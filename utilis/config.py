"""Configuration management system for K-Sparse AutoEncoder.

This module provides a flexible configuration system that supports:
- YAML and JSON configuration files
- Environment variable overrides
- Command-line argument integration
- Configuration validation and defaults
"""
import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import argparse


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""
    input_size: int = 784
    hidden_size: int = 100
    output_size: int = 784
    num_k_sparse: int = 25
    activation: str = "sigmoid"
    sparse_activation_type: str = "jump_relu"
    initialization_method: str = "tied"


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 1000
    learning_rate: float = 0.1
    batch_size: int = 64
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    print_epochs: int = 100
    
    # Optimizer settings
    optimizer: str = "sgd"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Learning rate scheduling
    lr_schedule: str = "constant"
    lr_decay_factor: float = 0.1
    lr_decay_patience: int = 5
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    gradient_clip_norm: Optional[float] = None


@dataclass
class LossConfig:
    """Configuration for loss function."""
    loss_type: str = "comprehensive_loss"
    mse_coeff: float = 1.0
    auxk_coeff: float = 0.02
    diversity_coeff: float = 0.01
    l1_coeff: float = 0.01
    dead_neuron_coeff: float = 0.001


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    enabled: bool = True
    initial_k_ratio: float = 0.6
    final_k_ratio: float = 1.0
    curriculum_epochs: int = 50


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset: str = "mnist"
    data_path: str = "data/"
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    normalize: bool = True
    augmentation: bool = False


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment."""
    name: str = "k_sparse_autoencoder"
    description: str = "K-Sparse AutoEncoder experiment"
    output_dir: str = "experiments/"
    random_seed: int = 42
    
    # Sub-configurations
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    data: DataConfig = field(default_factory=DataConfig)


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = ExperimentConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        format_type = self._detect_format(config_path)
        
        with open(config_path, 'r') as f:
            if format_type == ConfigFormat.JSON:
                config_dict = json.load(f)
            else:  # YAML
                config_dict = yaml.safe_load(f)
        
        self.config = self._dict_to_config(config_dict)
        return self.config
    
    def save_config(self, config_path: str, config: Optional[ExperimentConfig] = None) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration
            config: Configuration to save (uses current if None)
        """
        if config is None:
            config = self.config
        
        format_type = self._detect_format(config_path)
        config_dict = self._config_to_dict(config)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if format_type == ConfigFormat.JSON:
                json.dump(config_dict, f, indent=2)
            else:  # YAML
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update_from_env(self, prefix: str = "KSPARSE_") -> None:
        """Update configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables
        """
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        
        for env_key, env_value in env_vars.items():
            # Remove prefix and convert to lowercase
            config_key = env_key[len(prefix):].lower()
            
            # Parse nested keys (e.g., KSPARSE_TRAINING_EPOCHS)
            parts = config_key.split('_')
            
            if len(parts) >= 2:
                section = parts[0]
                key = '_'.join(parts[1:])
                
                # Convert string values to appropriate types
                value = self._parse_env_value(env_value)
                
                # Update configuration
                if hasattr(self.config, section):
                    section_config = getattr(self.config, section)
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        for key, value in vars(args).items():
            if value is not None:
                # Handle nested configuration keys
                if '.' in key:
                    section, attr = key.split('.', 1)
                    if hasattr(self.config, section):
                        section_config = getattr(self.config, section)
                        if hasattr(section_config, attr):
                            setattr(section_config, attr, value)
                else:
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
    
    def validate_config(self) -> bool:
        """Validate configuration values.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate network configuration
            assert self.config.network.input_size > 0
            assert self.config.network.hidden_size > 0
            assert self.config.network.output_size > 0
            assert 0 < self.config.network.num_k_sparse <= self.config.network.hidden_size
            
            # Validate training configuration
            assert self.config.training.epochs > 0
            assert self.config.training.learning_rate > 0
            assert self.config.training.batch_size > 0
            assert 0 <= self.config.training.validation_split < 1
            
            # Validate loss configuration
            assert self.config.loss.mse_coeff >= 0
            assert self.config.loss.auxk_coeff >= 0
            assert self.config.loss.diversity_coeff >= 0
            
            # Validate curriculum configuration
            if self.config.curriculum.enabled:
                assert 0 < self.config.curriculum.initial_k_ratio <= 1
                assert 0 < self.config.curriculum.final_k_ratio <= 1
                assert self.config.curriculum.curriculum_epochs > 0
            
            return True
            
        except AssertionError:
            return False
    
    def get_config_summary(self) -> str:
        """Get a summary of the current configuration.
        
        Returns:
            Configuration summary string
        """
        summary = []
        summary.append(f"Experiment: {self.config.name}")
        summary.append(f"Description: {self.config.description}")
        summary.append("")
        
        summary.append("Network Configuration:")
        summary.append(f"  Architecture: {self.config.network.input_size} -> {self.config.network.hidden_size} -> {self.config.network.output_size}")
        summary.append(f"  Sparsity (k): {self.config.network.num_k_sparse}")
        summary.append(f"  Activation: {self.config.network.activation}")
        summary.append(f"  Sparse Type: {self.config.network.sparse_activation_type}")
        summary.append("")
        
        summary.append("Training Configuration:")
        summary.append(f"  Epochs: {self.config.training.epochs}")
        summary.append(f"  Learning Rate: {self.config.training.learning_rate}")
        summary.append(f"  Batch Size: {self.config.training.batch_size}")
        summary.append(f"  Optimizer: {self.config.training.optimizer}")
        summary.append("")
        
        summary.append("Loss Configuration:")
        summary.append(f"  Type: {self.config.loss.loss_type}")
        summary.append(f"  Coefficients: MSE={self.config.loss.mse_coeff}, AuxK={self.config.loss.auxk_coeff}")
        summary.append("")
        
        if self.config.curriculum.enabled:
            summary.append("Curriculum Learning:")
            summary.append(f"  Initial k ratio: {self.config.curriculum.initial_k_ratio}")
            summary.append(f"  Final k ratio: {self.config.curriculum.final_k_ratio}")
            summary.append(f"  Curriculum epochs: {self.config.curriculum.curriculum_epochs}")
        
        return "\n".join(summary)
    
    def _detect_format(self, config_path: str) -> ConfigFormat:
        """Detect configuration file format from extension."""
        ext = os.path.splitext(config_path)[1].lower()
        if ext == '.json':
            return ConfigFormat.JSON
        elif ext in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            raise ValueError(f"Unsupported configuration format: {ext}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to configuration object."""
        config = ExperimentConfig()
        
        # Update main config fields
        for key, value in config_dict.items():
            if key in ['network', 'training', 'loss', 'curriculum', 'data']:
                # Handle nested configurations
                section_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(section_config, sub_key):
                        setattr(section_config, sub_key, sub_value)
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return asdict(config)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value


def create_default_config() -> ExperimentConfig:
    """Create a default configuration."""
    return ExperimentConfig()


def create_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create configuration from command-line arguments."""
    config_manager = ConfigManager()
    config_manager.update_from_args(args)
    return config_manager.config


def save_default_config(config_path: str = "config/default.yaml") -> None:
    """Save default configuration to file."""
    config_manager = ConfigManager()
    config_manager.save_config(config_path)


if __name__ == "__main__":
    # Create and save default configuration
    print("Creating default configuration...")
    save_default_config()
    print("Default configuration saved to config/default.yaml")
    
    # Load and display configuration
    manager = ConfigManager("config/default.yaml")
    print("\nConfiguration Summary:")
    print(manager.get_config_summary())