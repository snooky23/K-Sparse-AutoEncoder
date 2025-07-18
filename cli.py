"""Command-line interface for K-Sparse AutoEncoder.

This module provides a comprehensive CLI for training, evaluating, and experimenting
with K-Sparse AutoEncoders. It supports configuration files, environment variables,
and command-line arguments.
"""
import argparse
import sys
import os
import time
from typing import Optional, Dict, Any
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utilis.config import ConfigManager, ExperimentConfig
from utilis.loss_functions import LossType, LossFactory
from utilis.sparse_activations import SparseActivationType
from utilis.activations import sigmoid_function, relu_function, tanh_function
from utilis.mnist.mnist_helper import MnistHelper
from layers.improved_sparse_layer import ImprovedSparseLayer
from layers.linear_layer import LinearLayer
from nets.improved_fcnn import ImprovedFCNN


class KSparseAutoEncoderCLI:
    """Command-line interface for K-Sparse AutoEncoder."""
    
    def __init__(self):
        """Initialize CLI."""
        self.config_manager = ConfigManager()
        self.experiment_results = {}
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="K-Sparse AutoEncoder CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Train with default configuration
  python cli.py train
  
  # Train with custom configuration file
  python cli.py train --config config/experiment.yaml
  
  # Train with command-line overrides
  python cli.py train --network.num_k_sparse 30 --training.epochs 500
  
  # Evaluate a trained model
  python cli.py evaluate --model-path experiments/model.npz
  
  # Create default configuration file
  python cli.py create-config --output config/default.yaml
  
  # Run hyperparameter search
  python cli.py search --param network.num_k_sparse --values 10,20,30,40
            """
        )
        
        # Global arguments
        parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
        parser.add_argument('--output-dir', '-o', type=str, default='experiments/', help='Output directory')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
        parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train K-Sparse AutoEncoder')
        self._add_train_arguments(train_parser)
        
        # Evaluate command
        eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
        self._add_evaluate_arguments(eval_parser)
        
        # Create config command
        config_parser = subparsers.add_parser('create-config', help='Create configuration file')
        config_parser.add_argument('--output', type=str, default='config/default.yaml', 
                                 help='Output configuration file path')
        
        # Search command
        search_parser = subparsers.add_parser('search', help='Hyperparameter search')
        self._add_search_arguments(search_parser)
        
        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare different configurations')
        self._add_compare_arguments(compare_parser)
        
        return parser
    
    def _add_train_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add training-specific arguments."""
        # Network configuration
        net_group = parser.add_argument_group('Network Configuration')
        net_group.add_argument('--network.input-size', type=int, help='Input size')
        net_group.add_argument('--network.hidden-size', type=int, help='Hidden layer size')
        net_group.add_argument('--network.output-size', type=int, help='Output size')
        net_group.add_argument('--network.num-k-sparse', type=int, help='Number of sparse neurons')
        net_group.add_argument('--network.activation', type=str, choices=['sigmoid', 'relu', 'tanh'], 
                             help='Activation function')
        net_group.add_argument('--network.sparse-activation-type', type=str, 
                             choices=['hard_topk', 'jump_relu', 'gated_sparse', 'adaptive_sparse'],
                             help='Sparse activation type')
        net_group.add_argument('--network.initialization-method', type=str,
                             choices=['tied', 'xavier', 'he', 'sparse_friendly'],
                             help='Weight initialization method')
        
        # Training configuration
        train_group = parser.add_argument_group('Training Configuration')
        train_group.add_argument('--training.epochs', type=int, help='Number of training epochs')
        train_group.add_argument('--training.learning-rate', type=float, help='Learning rate')
        train_group.add_argument('--training.batch-size', type=int, help='Batch size')
        train_group.add_argument('--training.validation-split', type=float, help='Validation split')
        train_group.add_argument('--training.early-stopping-patience', type=int, help='Early stopping patience')
        train_group.add_argument('--training.optimizer', type=str, choices=['sgd', 'adam', 'rmsprop'],
                               help='Optimizer type')
        
        # Loss configuration
        loss_group = parser.add_argument_group('Loss Configuration')
        loss_group.add_argument('--loss.loss-type', type=str,
                              choices=['basic_mse', 'auxk_loss', 'diversity_loss', 'comprehensive_loss'],
                              help='Loss function type')
        loss_group.add_argument('--loss.mse-coeff', type=float, help='MSE coefficient')
        loss_group.add_argument('--loss.auxk-coeff', type=float, help='AuxK coefficient')
        loss_group.add_argument('--loss.diversity-coeff', type=float, help='Diversity coefficient')
        
        # Curriculum learning
        curriculum_group = parser.add_argument_group('Curriculum Learning')
        curriculum_group.add_argument('--curriculum.enabled', action='store_true', help='Enable curriculum learning')
        curriculum_group.add_argument('--curriculum.initial-k-ratio', type=float, help='Initial k ratio')
        curriculum_group.add_argument('--curriculum.final-k-ratio', type=float, help='Final k ratio')
        curriculum_group.add_argument('--curriculum.curriculum-epochs', type=int, help='Curriculum epochs')
        
        # Data configuration
        data_group = parser.add_argument_group('Data Configuration')
        data_group.add_argument('--data.dataset', type=str, choices=['mnist'], help='Dataset to use')
        data_group.add_argument('--data.train-samples', type=int, help='Number of training samples')
        data_group.add_argument('--data.test-samples', type=int, help='Number of test samples')
        
        # Output options
        parser.add_argument('--save-model', action='store_true', help='Save trained model')
        parser.add_argument('--save-history', action='store_true', help='Save training history')
        parser.add_argument('--save-config', action='store_true', help='Save used configuration')
    
    def _add_evaluate_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add evaluation-specific arguments."""
        parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
        parser.add_argument('--test-samples', type=int, help='Number of test samples')
        parser.add_argument('--visualize', action='store_true', help='Create visualizations')
        parser.add_argument('--save-results', action='store_true', help='Save evaluation results')
    
    def _add_search_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add hyperparameter search arguments."""
        parser.add_argument('--param', type=str, required=True, help='Parameter to search (e.g., network.num_k_sparse)')
        parser.add_argument('--values', type=str, required=True, help='Comma-separated values to try')
        parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'loss'], help='Metric to optimize')
        parser.add_argument('--trials', type=int, default=1, help='Number of trials per configuration')
    
    def _add_compare_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add comparison arguments."""
        parser.add_argument('--configs', type=str, nargs='+', required=True, 
                          help='Configuration files to compare')
        parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'loss'], 
                          help='Metric to compare')
        parser.add_argument('--visualize', action='store_true', help='Create comparison visualization')
    
    def run(self, args: Optional[list] = None) -> None:
        """Run the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Load configuration
        if parsed_args.config:
            self.config_manager.load_config(parsed_args.config)
        
        # Update from environment variables
        self.config_manager.update_from_env()
        
        # Update from command-line arguments
        self.config_manager.update_from_args(parsed_args)
        
        # Set global settings
        if hasattr(parsed_args, 'random_seed'):
            np.random.seed(parsed_args.random_seed)
        
        # Validate configuration
        if not self.config_manager.validate_config():
            print("❌ Configuration validation failed!")
            return
        
        # Execute command
        if parsed_args.command == 'train':
            self._run_training(parsed_args)
        elif parsed_args.command == 'evaluate':
            self._run_evaluation(parsed_args)
        elif parsed_args.command == 'create-config':
            self._create_config(parsed_args)
        elif parsed_args.command == 'search':
            self._run_search(parsed_args)
        elif parsed_args.command == 'compare':
            self._run_comparison(parsed_args)
        else:
            parser.print_help()
    
    def _run_training(self, args: argparse.Namespace) -> None:
        """Run training with current configuration."""
        config = self.config_manager.config
        
        print("🚀 Starting K-Sparse AutoEncoder Training")
        print("=" * 50)
        print(self.config_manager.get_config_summary())
        print("=" * 50)
        
        # Load data
        print("📚 Loading data...")
        data = self._load_data(config.data)
        
        # Create network
        print("🏗️  Creating network...")
        network = self._create_network(config)
        
        # Train network
        print("🎯 Training network...")
        start_time = time.time()
        
        history = network.train(
            data['train_x'], data['train_y'],
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            validation_split=config.training.validation_split,
            early_stopping_patience=config.training.early_stopping_patience,
            print_epochs=config.training.print_epochs
        )
        
        training_time = time.time() - start_time
        
        # Evaluate network
        print("📊 Evaluating network...")
        test_predictions = network.predict(data['test_x'])
        test_mse = np.mean((data['test_x'] - test_predictions) ** 2)
        
        # Display results
        print("\\n✅ Training Complete!")
        print(f"📈 Final Training Loss: {history['loss'][-1]:.4f}")
        print(f"📊 Test MSE: {test_mse:.4f}")
        print(f"⏱️  Training Time: {training_time:.1f}s")
        
        # Save results
        if args.save_model or args.save_history or args.save_config:
            self._save_results(network, history, config, args)
        
        # Store results for potential comparison
        self.experiment_results[config.name] = {
            'test_mse': test_mse,
            'final_loss': history['loss'][-1],
            'training_time': training_time,
            'config': config
        }
    
    def _run_evaluation(self, args: argparse.Namespace) -> None:
        """Run evaluation of a trained model."""
        print("🧪 Evaluating trained model...")
        # Implementation for model evaluation
        print("⚠️  Model evaluation not yet implemented")
    
    def _create_config(self, args: argparse.Namespace) -> None:
        """Create a default configuration file."""
        print(f"📝 Creating configuration file: {args.output}")
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save default configuration
        self.config_manager.save_config(args.output)
        
        print("✅ Configuration file created successfully!")
        print(f"📄 Edit {args.output} to customize your experiments")
    
    def _run_search(self, args: argparse.Namespace) -> None:
        """Run hyperparameter search."""
        print("🔍 Running hyperparameter search...")
        
        # Parse parameter and values
        param_path = args.param
        values = [self._parse_value(v.strip()) for v in args.values.split(',')]
        
        print(f"📊 Searching parameter: {param_path}")
        print(f"🎯 Values: {values}")
        print(f"🏁 Trials per value: {args.trials}")
        
        results = {}
        
        for value in values:
            print(f"\\n🧪 Testing {param_path} = {value}")
            
            trial_results = []
            for trial in range(args.trials):
                # Create modified configuration
                config = self._create_modified_config(param_path, value)
                
                # Run training
                result = self._run_single_experiment(config, verbose=False)
                trial_results.append(result[args.metric])
                
                if args.trials > 1:
                    print(f"  Trial {trial + 1}: {args.metric} = {result[args.metric]:.4f}")
            
            # Calculate statistics
            mean_result = np.mean(trial_results)
            std_result = np.std(trial_results)
            
            results[value] = {
                'mean': mean_result,
                'std': std_result,
                'trials': trial_results
            }
            
            print(f"  📊 {args.metric} = {mean_result:.4f} ± {std_result:.4f}")
        
        # Display final results
        print("\\n🏆 Search Results:")
        print(f"{'Value':<10} {'Mean':<10} {'Std':<10} {'Best':<10}")
        print("-" * 40)
        
        best_value = None
        best_score = float('inf') if args.metric == 'mse' else float('-inf')
        
        for value, result in results.items():
            is_best = (args.metric == 'mse' and result['mean'] < best_score) or \\
                     (args.metric == 'loss' and result['mean'] < best_score)
            
            if is_best:
                best_value = value
                best_score = result['mean']
            
            marker = "🥇" if is_best else "  "
            print(f"{marker}{value:<10} {result['mean']:<10.4f} {result['std']:<10.4f}")
        
        print(f"\\n🎯 Best value: {param_path} = {best_value}")
    
    def _run_comparison(self, args: argparse.Namespace) -> None:
        """Run comparison between different configurations."""
        print("⚖️  Comparing configurations...")
        
        results = {}
        
        for config_path in args.configs:
            print(f"\\n🧪 Testing configuration: {config_path}")
            
            # Load configuration
            config_manager = ConfigManager(config_path)
            config = config_manager.config
            
            # Run experiment
            result = self._run_single_experiment(config, verbose=False)
            results[config_path] = result
            
            print(f"  📊 {args.metric} = {result[args.metric]:.4f}")
        
        # Display comparison
        print("\\n📊 Comparison Results:")
        print(f"{'Configuration':<30} {args.metric.upper():<10}")
        print("-" * 40)
        
        for config_path, result in results.items():
            config_name = os.path.basename(config_path)
            print(f"{config_name:<30} {result[args.metric]:<10.4f}")
    
    def _load_data(self, data_config) -> Dict[str, np.ndarray]:
        """Load data based on configuration."""
        if data_config.dataset == 'mnist':
            mnist = MnistHelper()
            train_lbl, train_img, test_lbl, test_img = mnist.get_data()
            
            # Flatten and normalize
            train_x = train_img.reshape(-1, 784)
            test_x = test_img.reshape(-1, 784)
            
            if data_config.normalize:
                train_x = train_x / 255.0
                test_x = test_x / 255.0
            
            # Limit samples if specified
            if data_config.train_samples:
                train_x = train_x[:data_config.train_samples]
            if data_config.test_samples:
                test_x = test_x[:data_config.test_samples]
            
            return {
                'train_x': train_x,
                'train_y': train_x,  # Autoencoder: input = target
                'test_x': test_x,
                'test_y': test_x
            }
        else:
            raise ValueError(f"Unsupported dataset: {data_config.dataset}")
    
    def _create_network(self, config: ExperimentConfig) -> ImprovedFCNN:
        """Create network based on configuration."""
        # Map activation function names
        activation_map = {
            'sigmoid': sigmoid_function,
            'relu': relu_function,
            'tanh': tanh_function
        }
        
        activation_func = activation_map[config.network.activation]
        
        # Create decoder first for tied initialization
        decoder = LinearLayer(
            "decoder", 
            n_in=config.network.hidden_size,
            n_out=config.network.output_size,
            activation=activation_func
        )
        
        # Create encoder
        encoder = ImprovedSparseLayer(
            name="encoder",
            n_in=config.network.input_size,
            n_out=config.network.hidden_size,
            activation=activation_func,
            num_k_sparse=config.network.num_k_sparse,
            sparse_activation_type=SparseActivationType(config.network.sparse_activation_type),
            initialization_method=config.network.initialization_method,
            decoder_layer=decoder
        )
        
        # Create network
        network = ImprovedFCNN(
            layers=[encoder, decoder],
            loss_function=LossType(config.loss.loss_type),
            loss_config=asdict(config.loss),
            curriculum_learning=config.curriculum.enabled,
            curriculum_config=asdict(config.curriculum) if config.curriculum.enabled else None,
            dead_neuron_detection=True
        )
        
        return network
    
    def _run_single_experiment(self, config: ExperimentConfig, verbose: bool = True) -> Dict[str, Any]:
        """Run a single experiment with given configuration."""
        # Load data
        data = self._load_data(config.data)
        
        # Create network
        network = self._create_network(config)
        
        # Train network
        history = network.train(
            data['train_x'], data['train_y'],
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            validation_split=config.training.validation_split,
            early_stopping_patience=config.training.early_stopping_patience,
            print_epochs=config.training.print_epochs if verbose else 0
        )
        
        # Evaluate
        test_predictions = network.predict(data['test_x'])
        test_mse = np.mean((data['test_x'] - test_predictions) ** 2)
        
        return {
            'mse': test_mse,
            'loss': history['loss'][-1],
            'network': network,
            'history': history
        }
    
    def _create_modified_config(self, param_path: str, value: Any) -> ExperimentConfig:
        """Create a modified configuration with a parameter change."""
        config = ExperimentConfig()  # Start with default
        
        # Parse parameter path and set value
        parts = param_path.split('.')
        if len(parts) == 2:
            section, param = parts
            section_config = getattr(config, section)
            setattr(section_config, param.replace('-', '_'), value)
        
        return config
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value to appropriate type."""
        # Try boolean
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Try integer
        try:
            return int(value_str)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Return as string
        return value_str
    
    def _save_results(self, network: ImprovedFCNN, history: Dict, config: ExperimentConfig, args: argparse.Namespace) -> None:
        """Save training results."""
        # Create output directory
        output_dir = os.path.join(args.output_dir, config.name)
        os.makedirs(output_dir, exist_ok=True)
        
        if args.save_model:
            model_path = os.path.join(output_dir, 'model.npz')
            network.save_model(model_path)
            print(f"💾 Model saved to: {model_path}")
        
        if args.save_history:
            history_path = os.path.join(output_dir, 'history.json')
            import json
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"📈 History saved to: {history_path}")
        
        if args.save_config:
            config_path = os.path.join(output_dir, 'config.yaml')
            self.config_manager.save_config(config_path)
            print(f"⚙️  Configuration saved to: {config_path}")


def main():
    """Main CLI entry point."""
    cli = KSparseAutoEncoderCLI()
    cli.run()


if __name__ == "__main__":
    main()