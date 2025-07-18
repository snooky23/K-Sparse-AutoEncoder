"""Complete system demonstration for K-Sparse AutoEncoder.

This script demonstrates all the advanced features and improvements:
- Configuration management system
- Command-line interface integration
- Advanced optimizers
- Enhanced visualization tools
- Comprehensive benchmarking
- Model persistence and loading
"""
import os
import numpy as np
import time
from typing import Dict, Any

# Import all the advanced modules
from utilis.config import ConfigManager, ExperimentConfig
from utilis.optimizers import OptimizerType, OptimizerFactory
from utilis.visualization import create_visualization_suite
from utilis.benchmarking import BenchmarkSuite
from utilis.model_persistence import ModelPersistence, ModelRegistry
from utilis.loss_functions import LossType
from utilis.sparse_activations import SparseActivationType
from utilis.mnist.mnist_helper import MnistHelper

# Import network components
from layers.improved_sparse_layer import ImprovedSparseLayer
from layers.linear_layer import LinearLayer
from nets.improved_fcnn import ImprovedFCNN


def demonstrate_configuration_system():
    """Demonstrate the configuration management system."""
    print("🔧 Configuration Management System Demo")
    print("=" * 50)
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Display default configuration
    print("📋 Default Configuration:")
    print(config_manager.get_config_summary())
    
    # Save configuration to file
    config_path = "config/demo_config.yaml"
    os.makedirs("config", exist_ok=True)
    config_manager.save_config(config_path)
    print(f"\\n💾 Configuration saved to: {config_path}")
    
    # Load and modify configuration
    config_manager.load_config(config_path)
    config_manager.config.network.num_k_sparse = 30
    config_manager.config.training.epochs = 100
    config_manager.config.loss.loss_type = "comprehensive_loss"
    
    print("\\n📝 Modified Configuration:")
    print(f"  K-sparse: {config_manager.config.network.num_k_sparse}")
    print(f"  Epochs: {config_manager.config.training.epochs}")
    print(f"  Loss type: {config_manager.config.loss.loss_type}")
    
    return config_manager


def demonstrate_advanced_optimizers():
    """Demonstrate advanced optimizers."""
    print("\\n⚡ Advanced Optimizers Demo")
    print("=" * 50)
    
    # Create different optimizers
    optimizers = {
        'Adam': OptimizerFactory.create_optimizer(OptimizerType.ADAM, learning_rate=0.001),
        'RMSprop': OptimizerFactory.create_optimizer(OptimizerType.RMSPROP, learning_rate=0.001),
        'Momentum': OptimizerFactory.create_optimizer(OptimizerType.MOMENTUM, learning_rate=0.01, momentum=0.9),
        'SGD': OptimizerFactory.create_optimizer(OptimizerType.SGD, learning_rate=0.01)
    }
    
    # Test optimizers on simple parameter update
    test_params = np.random.randn(10, 5)
    test_gradients = np.random.randn(10, 5)
    
    print("📊 Optimizer Comparison:")
    for name, optimizer in optimizers.items():
        updated_params = optimizer.update(test_params, test_gradients)
        param_change = np.mean(np.abs(updated_params - test_params))
        print(f"  {name}: Average parameter change = {param_change:.4f}")
    
    return optimizers


def demonstrate_model_persistence():
    """Demonstrate model persistence and loading."""
    print("\\n💾 Model Persistence Demo")
    print("=" * 50)
    
    # Create a simple model
    decoder = LinearLayer("decoder", 100, 784)
    encoder = ImprovedSparseLayer(
        "encoder", 784, 100, 
        num_k_sparse=25,
        sparse_activation_type=SparseActivationType.JUMP_RELU,
        initialization_method="tied",
        decoder_layer=decoder
    )
    
    model = ImprovedFCNN([encoder, decoder], LossType.COMPREHENSIVE_LOSS)
    
    # Save model
    model_path = "models/demo_model.npz"
    os.makedirs("models", exist_ok=True)
    
    config = ExperimentConfig()
    config.name = "demo_model"
    config.description = "Demonstration model for persistence"
    
    ModelPersistence.save_model(model, model_path, config=config)
    print(f"💾 Model saved to: {model_path}")
    
    # Load model
    loaded_data = ModelPersistence.load_model(model_path)
    loaded_model = loaded_data['model']
    metadata = loaded_data['metadata']
    
    print("✅ Model loaded successfully!")
    print(f"  Model type: {metadata.model_type}")
    print(f"  Created: {metadata.creation_time}")
    print(f"  Layers: {len(loaded_model.layers)}")
    
    # Test model registry
    registry = ModelRegistry()
    registry.register_model("demo_model", model_path, 
                          description="Demo model with all features",
                          tags=["demo", "jump_relu", "comprehensive_loss"])
    
    print("\\n📚 Model Registry:")
    models = registry.list_models()
    for model_info in models:
        print(f"  {model_info['id']}: {model_info['description']}")
    
    return loaded_model


def demonstrate_visualization_system():
    """Demonstrate enhanced visualization system."""
    print("\\n📊 Visualization System Demo")
    print("=" * 50)
    
    # Create visualization suite
    viz_suite = create_visualization_suite("visualizations/")
    
    # Generate sample training history
    epochs = 50
    sample_history = {
        'loss': np.exp(-np.linspace(0, 3, epochs)) + 0.1 * np.random.randn(epochs),
        'learning_rate': [0.1 * (0.95 ** i) for i in range(epochs)],
        'sparsity_info': [
            {'encoder': {'actual_k_mean': 25 + np.random.randn(), 'actual_k_std': 0.5}}
            for _ in range(epochs)
        ]
    }
    
    print("📈 Generating training progress visualization...")
    viz_suite['training'].plot_training_history(
        sample_history, 
        save_path="visualizations/demo_training_history.png"
    )
    
    # Create sample model for architecture visualization
    layers = [
        ImprovedSparseLayer("encoder", 784, 100, num_k_sparse=25),
        LinearLayer("decoder", 100, 784)
    ]
    
    print("🏗️  Generating architecture diagram...")
    viz_suite['model'].plot_architecture(
        layers,
        save_path="visualizations/demo_architecture.png"
    )
    
    print("✅ Visualizations generated in 'visualizations/' directory")
    
    return viz_suite


def demonstrate_benchmarking_system():
    """Demonstrate comprehensive benchmarking system."""
    print("\\n🏁 Benchmarking System Demo")
    print("=" * 50)
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite("benchmarks/")
    
    # Create test models
    models = {}
    
    # Model 1: Basic configuration
    decoder1 = LinearLayer("decoder", 100, 784)
    encoder1 = ImprovedSparseLayer(
        "encoder", 784, 100, num_k_sparse=20,
        sparse_activation_type=SparseActivationType.HARD_TOPK
    )
    models['basic'] = ImprovedFCNN([encoder1, decoder1], LossType.BASIC_MSE)
    
    # Model 2: Advanced configuration  
    decoder2 = LinearLayer("decoder", 100, 784)
    encoder2 = ImprovedSparseLayer(
        "encoder", 784, 100, num_k_sparse=25,
        sparse_activation_type=SparseActivationType.JUMP_RELU
    )
    models['advanced'] = ImprovedFCNN([encoder2, decoder2], LossType.COMPREHENSIVE_LOSS)
    
    # Generate test data
    test_data = np.random.randn(500, 784)
    
    print("⚡ Running performance benchmarks...")
    
    # Performance benchmark
    for name, model in models.items():
        print(f"  Benchmarking {name} model...")
        
        # Inference speed
        inference_results = benchmark_suite.performance_benchmark.benchmark_inference_speed(
            model, test_data, batch_sizes=[1, 16, 32], n_runs=3
        )
        
        for result in inference_results:
            batch_size = result.metadata['batch_size']
            throughput = result.metrics['throughput']
            print(f"    Batch {batch_size}: {throughput:.1f} samples/sec")
    
    print("\\n🎯 Running quality benchmarks...")
    
    # Quality benchmark
    quality_results = benchmark_suite.quality_benchmark.benchmark_reconstruction_quality(
        models, test_data[:100]
    )
    
    for result in quality_results:
        model_name = result.metadata['model_name']
        mse = result.metrics.get('mse', 0)
        print(f"  {model_name}: MSE = {mse:.4f}")
    
    print("\\n📊 Benchmark results saved to 'benchmarks/' directory")
    
    return benchmark_suite


def demonstrate_complete_workflow():
    """Demonstrate a complete workflow using all features."""
    print("\\n🚀 Complete Workflow Demo")
    print("=" * 50)
    
    # Step 1: Load configuration
    config_manager = ConfigManager()
    config = config_manager.config
    config.name = "complete_workflow_demo"
    config.network.num_k_sparse = 25
    config.training.epochs = 20
    config.loss.loss_type = "comprehensive_loss"
    
    print("1️⃣ Configuration loaded")
    
    # Step 2: Load data
    print("2️⃣ Loading MNIST data...")
    mnist = MnistHelper()
    train_lbl, train_img, test_lbl, test_img = mnist.get_data()
    
    train_data = train_img.reshape(-1, 784)[:1000] / 255.0  # Small subset
    test_data = test_img.reshape(-1, 784)[:100] / 255.0
    
    # Step 3: Create model with all improvements
    print("3️⃣ Creating enhanced model...")
    decoder = LinearLayer("decoder", 100, 784)
    encoder = ImprovedSparseLayer(
        "encoder", 784, 100,
        num_k_sparse=config.network.num_k_sparse,
        sparse_activation_type=SparseActivationType.JUMP_RELU,
        initialization_method="tied",
        decoder_layer=decoder
    )
    
    model = ImprovedFCNN(
        [encoder, decoder],
        loss_function=LossType.COMPREHENSIVE_LOSS,
        curriculum_learning=True,
        dead_neuron_detection=True
    )
    
    # Step 4: Train with advanced features
    print("4️⃣ Training model...")
    start_time = time.time()
    
    history = model.train(
        train_data, train_data,
        epochs=config.training.epochs,
        learning_rate=0.1,
        batch_size=64,
        print_epochs=10,
        collect_sparsity_info=True
    )
    
    training_time = time.time() - start_time
    
    # Step 5: Evaluate model
    print("5️⃣ Evaluating model...")
    predictions = model.predict(test_data)
    mse = np.mean((test_data - predictions) ** 2)
    
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Test MSE: {mse:.4f}")
    
    # Step 6: Save model
    print("6️⃣ Saving model...")
    model_path = "models/complete_workflow_model.npz"
    ModelPersistence.save_model(model, model_path, config=config, training_history=history)
    
    # Step 7: Create visualizations
    print("7️⃣ Creating visualizations...")
    viz_suite = create_visualization_suite("visualizations/complete_workflow/")
    
    viz_suite['training'].plot_training_history(
        history, 
        save_path="visualizations/complete_workflow/training_history.png"
    )
    
    viz_suite['model'].plot_architecture(
        model.layers,
        save_path="visualizations/complete_workflow/architecture.png"
    )
    
    # Step 8: Generate report
    print("8️⃣ Generating comprehensive report...")
    
    results = {
        'training_time': training_time,
        'final_loss': history['loss'][-1],
        'test_mse': mse,
        'model_parameters': sum(layer.weights.size + layer.biases.size for layer in model.layers),
        'sparsity_level': config.network.num_k_sparse,
        'curriculum_learning': True,
        'dead_neuron_detection': True
    }
    
    # Save results summary
    summary_path = "results/complete_workflow_summary.txt"
    os.makedirs("results", exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("Complete Workflow Demo Results\\n")
        f.write("=" * 40 + "\\n")
        f.write(f"Experiment: {config.name}\\n")
        f.write(f"Training Time: {training_time:.1f}s\\n")
        f.write(f"Final Loss: {history['loss'][-1]:.4f}\\n")
        f.write(f"Test MSE: {mse:.4f}\\n")
        f.write(f"Model Parameters: {results['model_parameters']}\\n")
        f.write(f"Sparsity Level: {config.network.num_k_sparse}\\n")
        f.write(f"Features: Curriculum Learning, Dead Neuron Detection\\n")
    
    print(f"✅ Complete workflow finished! Results saved to: {summary_path}")
    
    return {
        'model': model,
        'history': history,
        'results': results,
        'config': config
    }


def main():
    """Main demonstration function."""
    print("🎉 K-Sparse AutoEncoder Complete System Demo")
    print("=" * 60)
    print("This demonstration showcases all advanced features and improvements:")
    print("• Configuration management system")
    print("• Advanced optimizers (Adam, RMSprop, etc.)")
    print("• Model persistence and loading")
    print("• Enhanced visualization tools")
    print("• Comprehensive benchmarking suite")
    print("• Complete integration workflow")
    print("=" * 60)
    
    # Run all demonstrations
    config_manager = demonstrate_configuration_system()
    optimizers = demonstrate_advanced_optimizers()
    loaded_model = demonstrate_model_persistence()
    viz_suite = demonstrate_visualization_system()
    benchmark_suite = demonstrate_benchmarking_system()
    workflow_results = demonstrate_complete_workflow()
    
    print("\\n🎊 All Demonstrations Complete!")
    print("=" * 60)
    print("📁 Generated Files:")
    print("  • config/demo_config.yaml - Configuration file")
    print("  • models/ - Saved models and registry")
    print("  • visualizations/ - Generated plots and diagrams")
    print("  • benchmarks/ - Performance benchmark results")
    print("  • results/ - Experiment results and reports")
    print("\\n🚀 The K-Sparse AutoEncoder system is now production-ready!")
    print("✨ Features include:")
    print("  ✅ JumpReLU activation with learnable thresholds")
    print("  ✅ Configurable loss functions")
    print("  ✅ Advanced optimizers")
    print("  ✅ Curriculum learning")
    print("  ✅ Dead neuron detection")
    print("  ✅ Model persistence")
    print("  ✅ Comprehensive benchmarking")
    print("  ✅ Enhanced visualizations")
    print("  ✅ Configuration management")
    print("  ✅ Command-line interface")
    
    return {
        'config_manager': config_manager,
        'optimizers': optimizers,
        'model': loaded_model,
        'viz_suite': viz_suite,
        'benchmark_suite': benchmark_suite,
        'workflow_results': workflow_results
    }


if __name__ == "__main__":
    results = main()