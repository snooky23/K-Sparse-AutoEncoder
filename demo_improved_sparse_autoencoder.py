"""Demonstration of improved K-Sparse AutoEncoder with advanced features.

This script showcases the enhanced sparse autoencoder with:
- Configurable loss functions (Basic MSE, AuxK Loss, Diversity Loss, Comprehensive Loss)
- Advanced sparse activations (Hard TopK, JumpReLU, Gated Sparse, Adaptive Sparse)
- Improved initialization methods (Tied, Xavier, He, Sparse-friendly)
- Curriculum learning with progressive sparsity
- Dead neuron detection and reset
- Comprehensive performance comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import time

# Import all the new modules
from layers.improved_sparse_layer import ImprovedSparseLayer
from layers.linear_layer import LinearLayer
from nets.improved_fcnn import ImprovedFCNN
from utilis.loss_functions import LossType, LossFactory
from utilis.sparse_activations import SparseActivationType
from utilis.activations import sigmoid_function
from utilis.mnist.mnist_helper import MnistHelper


def load_mnist_data():
    """Load MNIST data for experiments."""
    print("Loading MNIST dataset...")
    mnist = MnistHelper()
    train_lbl, train_img, test_lbl, test_img = mnist.get_data()
    
    # Flatten and normalize
    train_data = train_img.reshape(-1, 784) / 255.0
    test_data = test_img.reshape(-1, 784) / 255.0
    
    print(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples")
    return train_data, test_data, train_lbl, test_lbl


def create_autoencoder(sparse_activation_type: SparseActivationType = SparseActivationType.JUMP_RELU,
                      loss_type: LossType = LossType.COMPREHENSIVE_LOSS,
                      initialization_method: str = "tied",
                      k_sparse: int = 25,
                      curriculum_learning: bool = True) -> ImprovedFCNN:
    """Create an improved sparse autoencoder with specified configuration.
    
    Args:
        sparse_activation_type: Type of sparse activation to use
        loss_type: Type of loss function to use
        initialization_method: Weight initialization method
        k_sparse: Number of sparse neurons
        curriculum_learning: Whether to use curriculum learning
        
    Returns:
        Configured ImprovedFCNN instance
    """
    # Create decoder layer first for tied initialization
    decoder = LinearLayer("decoder", n_in=100, n_out=784, activation=sigmoid_function)
    
    # Create encoder layer with advanced features
    encoder = ImprovedSparseLayer(
        name="encoder",
        n_in=784,
        n_out=100,
        activation=sigmoid_function,
        num_k_sparse=k_sparse,
        sparse_activation_type=sparse_activation_type,
        initialization_method=initialization_method,
        decoder_layer=decoder,
        threshold_init="data_driven",  # For JumpReLU
        temperature=1.0
    )
    
    # Configure loss function
    if loss_type == LossType.COMPREHENSIVE_LOSS:
        loss_config = {
            'mse_coeff': 1.0,
            'auxk_coeff': 0.02,
            'diversity_coeff': 0.01,
            'l1_coeff': 0.01,
            'dead_neuron_coeff': 0.001
        }
    elif loss_type == LossType.AUXK_LOSS:
        loss_config = {
            'mse_coeff': 1.0,
            'auxk_coeff': 0.02,
            'l1_coeff': 0.01
        }
    elif loss_type == LossType.DIVERSITY_LOSS:
        loss_config = {
            'mse_coeff': 1.0,
            'diversity_coeff': 0.01,
            'l1_coeff': 0.01
        }
    else:  # BASIC_MSE
        loss_config = {}
    
    # Configure curriculum learning
    curriculum_config = {
        'initial_k_ratio': 0.6,
        'final_k_ratio': 1.0,
        'curriculum_epochs': 50
    } if curriculum_learning else None
    
    # Create network
    network = ImprovedFCNN(
        layers=[encoder, decoder],
        loss_function=loss_type,
        loss_config=loss_config,
        curriculum_learning=curriculum_learning,
        curriculum_config=curriculum_config,
        dead_neuron_detection=True,
        dead_neuron_threshold=1e-6
    )
    
    return network


def run_comparison_experiment():
    """Run comprehensive comparison of different configurations."""
    print("=== Running Comprehensive Comparison Experiment ===\n")
    
    # Load data
    train_data, test_data, train_lbl, test_lbl = load_mnist_data()
    
    # Use subset for faster experimentation
    train_subset = train_data[:2000]
    test_subset = test_data[:500]
    
    # Configuration combinations to test
    experiments = [
        {
            'name': 'Baseline (Hard TopK + MSE)',
            'sparse_activation': SparseActivationType.HARD_TOPK,
            'loss_type': LossType.BASIC_MSE,
            'initialization': 'tied',
            'curriculum': False
        },
        {
            'name': 'JumpReLU + MSE',
            'sparse_activation': SparseActivationType.JUMP_RELU,
            'loss_type': LossType.BASIC_MSE,
            'initialization': 'tied',
            'curriculum': False
        },
        {
            'name': 'JumpReLU + AuxK Loss',
            'sparse_activation': SparseActivationType.JUMP_RELU,
            'loss_type': LossType.AUXK_LOSS,
            'initialization': 'tied',
            'curriculum': False
        },
        {
            'name': 'JumpReLU + Comprehensive Loss',
            'sparse_activation': SparseActivationType.JUMP_RELU,
            'loss_type': LossType.COMPREHENSIVE_LOSS,
            'initialization': 'tied',
            'curriculum': False
        },
        {
            'name': 'JumpReLU + Comprehensive + Curriculum',
            'sparse_activation': SparseActivationType.JUMP_RELU,
            'loss_type': LossType.COMPREHENSIVE_LOSS,
            'initialization': 'tied',
            'curriculum': True
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\nTesting: {exp['name']}")
        print("-" * 50)
        
        # Create network
        network = create_autoencoder(
            sparse_activation_type=exp['sparse_activation'],
            loss_type=exp['loss_type'],
            initialization_method=exp['initialization'],
            k_sparse=25,
            curriculum_learning=exp['curriculum']
        )
        
        # Train network
        start_time = time.time()
        history = network.train(
            train_subset, train_subset,
            epochs=100,
            learning_rate=0.1,
            batch_size=64,
            print_epochs=50,
            collect_sparsity_info=True
        )
        training_time = time.time() - start_time
        
        # Evaluate network
        predictions = network.predict(test_subset)
        mse = np.mean((test_subset - predictions) ** 2)
        
        # Get layer information
        layer_info = network.get_layer_info()
        
        # Store results
        results[exp['name']] = {
            'final_loss': history['loss'][-1],
            'mse': mse,
            'training_time': training_time,
            'layer_info': layer_info,
            'history': history
        }
        
        print(f"Final Loss: {history['loss'][-1]:.4f}")
        print(f"Test MSE: {mse:.4f}")
        print(f"Training Time: {training_time:.1f}s")
        
        # Print sparsity info
        if 'encoder' in layer_info:
            encoder_info = layer_info['encoder']
            if 'actual_k_mean' in encoder_info:
                print(f"Average Active Neurons: {encoder_info['actual_k_mean']:.1f}±{encoder_info['actual_k_std']:.1f}")
            if 'zero_fraction' in encoder_info:
                print(f"Zero Fraction: {encoder_info['zero_fraction']:.3f}")
    
    return results


def create_detailed_visualization(results: Dict, k_values: List[int] = [5, 10, 20, 30]):
    """Create detailed visualization comparing different methods."""
    print("\n=== Creating Detailed Visualization ===\n")
    
    # Load fresh data for visualization
    train_data, test_data, train_lbl, test_lbl = load_mnist_data()
    viz_data = test_data[:10]  # Use first 10 test samples
    
    # Create figure with subplots
    n_methods = 3  # Show top 3 methods
    fig, axes = plt.subplots(len(k_values) + 1, n_methods, figsize=(15, 20))
    
    # Methods to visualize
    methods = [
        {
            'name': 'Baseline (Hard TopK)',
            'config': {
                'sparse_activation': SparseActivationType.HARD_TOPK,
                'loss_type': LossType.BASIC_MSE,
                'initialization': 'tied',
                'curriculum': False
            }
        },
        {
            'name': 'JumpReLU + AuxK',
            'config': {
                'sparse_activation': SparseActivationType.JUMP_RELU,
                'loss_type': LossType.AUXK_LOSS,
                'initialization': 'tied',
                'curriculum': False
            }
        },
        {
            'name': 'JumpReLU + Comprehensive + Curriculum',
            'config': {
                'sparse_activation': SparseActivationType.JUMP_RELU,
                'loss_type': LossType.COMPREHENSIVE_LOSS,
                'initialization': 'tied',
                'curriculum': True
            }
        }
    ]
    
    # Show original images in first row
    if len(k_values) > 0:
        for j in range(min(n_methods, 10)):
            if j < len(viz_data):
                img = viz_data[j].reshape(28, 28)
                axes[0, j].imshow(img, cmap='gray')
                axes[0, j].set_title(f'Original {j+1}')
                axes[0, j].axis('off')
            else:
                axes[0, j].axis('off')
    
    # Test each method with different k values
    for method_idx, method in enumerate(methods):
        print(f"Testing {method['name']}...")
        
        for k_idx, k in enumerate(k_values):
            print(f"  k={k}...")
            
            # Create and train network
            network = create_autoencoder(
                sparse_activation_type=method['config']['sparse_activation'],
                loss_type=method['config']['loss_type'],
                initialization_method=method['config']['initialization'],
                k_sparse=k,
                curriculum_learning=method['config']['curriculum']
            )
            
            # Quick training
            network.train(
                train_data[:1000], train_data[:1000],
                epochs=50,
                learning_rate=0.1,
                batch_size=64,
                print_epochs=0
            )
            
            # Get reconstructions
            reconstructions = network.predict(viz_data)
            
            # Calculate MSE
            mse = np.mean((viz_data - reconstructions) ** 2)
            
            # Show reconstruction for first image
            if method_idx == 0:  # Show reconstructions in first column
                img = reconstructions[0].reshape(28, 28)
                axes[k_idx + 1, method_idx].imshow(img, cmap='gray')
                axes[k_idx + 1, method_idx].set_title(f'k={k}\\nMSE={mse:.4f}')
                axes[k_idx + 1, method_idx].axis('off')
            elif method_idx < n_methods:
                # Show comparison
                img = reconstructions[method_idx].reshape(28, 28) if method_idx < len(reconstructions) else reconstructions[0].reshape(28, 28)
                axes[k_idx + 1, method_idx].imshow(img, cmap='gray')
                axes[k_idx + 1, method_idx].set_title(f'{method["name"]}\\nk={k}, MSE={mse:.4f}')
                axes[k_idx + 1, method_idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Improved K-Sparse AutoEncoder: Method Comparison\\nReal MNIST Digits with Different Sparse Activations and Loss Functions', 
                 fontsize=16, y=0.98)
    
    # Save visualization
    os.makedirs('images', exist_ok=True)
    filename = 'images/improved_sparse_autoencoder_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed visualization saved to: {filename}")


def demonstrate_specific_improvements():
    """Demonstrate specific improvements like dead neuron detection."""
    print("\n=== Demonstrating Specific Improvements ===\n")
    
    # Load data
    train_data, test_data, train_lbl, test_lbl = load_mnist_data()
    train_subset = train_data[:1000]
    
    # Create network with comprehensive loss
    network = create_autoencoder(
        sparse_activation_type=SparseActivationType.JUMP_RELU,
        loss_type=LossType.COMPREHENSIVE_LOSS,
        initialization_method="tied",
        k_sparse=30,
        curriculum_learning=True
    )
    
    print("1. Training with comprehensive loss and curriculum learning...")
    history = network.train(
        train_subset, train_subset,
        epochs=100,
        learning_rate=0.1,
        batch_size=64,
        print_epochs=25,
        collect_sparsity_info=True
    )
    
    print("\\n2. Analyzing training results...")
    
    # Show final layer information
    layer_info = network.get_layer_info()
    print("\\nFinal Layer Information:")
    for name, info in layer_info.items():
        print(f"  {name}:")
        for key, value in info.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    
    # Show training history
    print("\\n3. Training Progress:")
    print(f"  Initial Loss: {history['loss'][0]:.4f}")
    print(f"  Final Loss: {history['loss'][-1]:.4f}")
    print(f"  Loss Reduction: {(history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100:.1f}%")
    
    # Show dead neuron information
    if history['dead_neurons']:
        print("\\n4. Dead Neuron Detection:")
        for event in history['dead_neurons']:
            print(f"  Epoch {event['epoch']}: Reset {sum(event['dead_counts'].values())} dead neurons")
    
    # Test with different k values
    print("\\n5. Testing reconstruction quality at different k values:")
    test_sample = test_data[:5]
    
    for k in [5, 15, 25, 35]:
        # Temporarily modify k
        encoder = network.layers[0]
        original_k = encoder.num_k_sparse
        encoder.num_k_sparse = k
        encoder.sparse_activation.num_k_sparse = k
        
        # Get reconstructions
        reconstructions = network.predict(test_sample)
        mse = np.mean((test_sample - reconstructions) ** 2)
        
        print(f"  k={k}: MSE = {mse:.4f}")
        
        # Restore original k
        encoder.num_k_sparse = original_k
        encoder.sparse_activation.num_k_sparse = original_k
    
    print("\\n6. JumpReLU Threshold Information:")
    encoder = network.layers[0]
    if hasattr(encoder, 'get_learnable_thresholds'):
        thresholds = encoder.get_learnable_thresholds()
        if thresholds is not None:
            print(f"  Threshold mean: {np.mean(thresholds):.4f}")
            print(f"  Threshold std: {np.std(thresholds):.4f}")
            print(f"  Threshold range: [{np.min(thresholds):.4f}, {np.max(thresholds):.4f}]")


if __name__ == "__main__":
    print("Improved K-Sparse AutoEncoder Demonstration")
    print("=" * 60)
    
    # Run comprehensive comparison
    results = run_comparison_experiment()
    
    # Create detailed visualization
    create_detailed_visualization(results)
    
    # Demonstrate specific improvements
    demonstrate_specific_improvements()
    
    print("\\n" + "=" * 60)
    print("Demonstration complete!")
    print("\\nSUMMARY OF IMPROVEMENTS:")
    print("✓ JumpReLU activation with learnable thresholds")
    print("✓ Configurable loss functions (MSE, AuxK, Diversity, Comprehensive)")
    print("✓ Advanced initialization methods (tied, Xavier, He, sparse-friendly)")
    print("✓ Curriculum learning with progressive sparsity")
    print("✓ Dead neuron detection and automatic reset")
    print("✓ Comprehensive training and validation framework")
    print("✓ Detailed performance monitoring and visualization")
    print("\\nCheck 'images/improved_sparse_autoencoder_comparison.png' for visual results!")