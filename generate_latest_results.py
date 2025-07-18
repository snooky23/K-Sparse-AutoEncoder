#!/usr/bin/env python3
"""
Generate latest results with improved algorithm and all k values including k=50
"""
import numpy as np
import matplotlib.pyplot as plt
from utilis.mnist.mnist_helper import MnistHelper
from layers.improved_sparse_layer import ImprovedSparseLayer
from layers.linear_layer import LinearLayer
from nets.improved_fcnn import ImprovedFCNN
from utilis.loss_functions import LossType
from utilis.sparse_activations import SparseActivationType
import time
import os

def create_improved_model(k_sparse=25):
    """Create improved model with JumpReLU and comprehensive loss."""
    # Create decoder first for tied weights
    decoder = LinearLayer("decoder", 100, 784)
    
    # Create encoder with improved features
    encoder = ImprovedSparseLayer(
        "encoder", 784, 100, 
        num_k_sparse=k_sparse,
        sparse_activation_type=SparseActivationType.JUMP_RELU,
        initialization_method="tied",
        decoder_layer=decoder
    )
    
    # Create improved network
    model = ImprovedFCNN(
        [encoder, decoder],
        loss_function=LossType.COMPREHENSIVE_LOSS,
        curriculum_learning=True,
        dead_neuron_detection=True
    )
    
    return model

def main():
    print("🚀 Generating Latest Results with Improved Algorithm")
    print("=" * 60)
    
    # Load MNIST data
    print("📊 Loading MNIST data...")
    mnist = MnistHelper()
    train_lbl, train_img, test_lbl, test_img = mnist.get_data()
    
    # Prepare data
    train_data = train_img.reshape(-1, 784)[:5000] / 255.0  # Use subset for faster training
    test_data = test_img.reshape(-1, 784)[:1000] / 255.0
    
    # Test different k values including k=50
    k_values = [5, 10, 20, 30, 50]
    results = {}
    
    print("🧠 Training models with different k values...")
    
    for k in k_values:
        print(f"\n🔹 Training with k={k}...")
        
        # Create improved model
        model = create_improved_model(k_sparse=k)
        
        # Train model
        start_time = time.time()
        history = model.train(
            train_data, train_data,
            epochs=30,
            learning_rate=0.1,
            batch_size=64,
            print_epochs=10
        )
        training_time = time.time() - start_time
        
        # Evaluate on test data
        predictions = model.predict(test_data)
        mse = np.mean((test_data - predictions) ** 2)
        
        results[k] = {
            'mse': mse,
            'final_loss': history['loss'][-1],
            'training_time': training_time,
            'predictions': predictions[:10]  # Save first 10 for visualization
        }
        
        print(f"  ✅ k={k}: MSE={mse:.4f}, Loss={history['loss'][-1]:.4f}, Time={training_time:.1f}s")
    
    # Create visualization
    print("\n📊 Creating comprehensive visualization...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance comparison
    ax1 = plt.subplot(3, 4, 1)
    k_list = list(results.keys())
    mse_list = [results[k]['mse'] for k in k_list]
    
    plt.plot(k_list, mse_list, 'bo-', linewidth=2, markersize=8)
    plt.title('Reconstruction Quality vs Sparsity\n(Improved Algorithm)', fontsize=12, fontweight='bold')
    plt.xlabel('k (Active Neurons)')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    
    # 2. Training time comparison
    ax2 = plt.subplot(3, 4, 2)
    time_list = [results[k]['training_time'] for k in k_list]
    plt.bar(k_list, time_list, alpha=0.7, color='orange')
    plt.title('Training Time vs Sparsity', fontsize=12, fontweight='bold')
    plt.xlabel('k (Active Neurons)')
    plt.ylabel('Training Time (s)')
    plt.xticks(k_list)
    
    # 3. Loss comparison
    ax3 = plt.subplot(3, 4, 3)
    loss_list = [results[k]['final_loss'] for k in k_list]
    plt.bar(k_list, loss_list, alpha=0.7, color='green')
    plt.title('Final Loss vs Sparsity', fontsize=12, fontweight='bold')
    plt.xlabel('k (Active Neurons)')
    plt.ylabel('Final Loss')
    plt.xticks(k_list)
    
    # 4. Sparsity vs Compression
    ax4 = plt.subplot(3, 4, 4)
    compression_ratios = [1 - (k/100) for k in k_list]
    plt.plot(k_list, compression_ratios, 'ro-', linewidth=2, markersize=8)
    plt.title('Compression Ratio vs k', fontsize=12, fontweight='bold')
    plt.xlabel('k (Active Neurons)')
    plt.ylabel('Compression Ratio')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    
    # 5-8. Reconstruction samples for each k value
    sample_indices = [0, 1, 2, 3]  # Show 4 samples
    
    for i, k in enumerate([5, 10, 30, 50]):  # Show key k values
        ax = plt.subplot(3, 4, 5 + i)
        
        # Show original and reconstruction side by side
        sample_idx = 0
        original = test_data[sample_idx].reshape(28, 28)
        reconstructed = results[k]['predictions'][sample_idx].reshape(28, 28)
        
        # Create side-by-side comparison
        comparison = np.hstack([original, reconstructed])
        plt.imshow(comparison, cmap='gray')
        plt.title(f'k={k} (MSE={results[k]["mse"]:.4f})', fontsize=10, fontweight='bold')
        plt.axis('off')
    
    # 9-12. Show multiple samples for k=30 (the problem case)
    for i in range(4):
        ax = plt.subplot(3, 4, 9 + i)
        sample_idx = i
        original = test_data[sample_idx].reshape(28, 28)
        reconstructed = results[30]['predictions'][sample_idx].reshape(28, 28)
        
        # Create side-by-side comparison
        comparison = np.hstack([original, reconstructed])
        plt.imshow(comparison, cmap='gray')
        plt.title(f'k=30 Sample {i+1}\n(Improved Algorithm)', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comprehensive results
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/comprehensive_k_sparse_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate detailed reconstruction comparison
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    
    for k_idx, k in enumerate(k_values):
        for sample_idx in range(10):
            if sample_idx < len(results[k]['predictions']):
                original = test_data[sample_idx].reshape(28, 28)
                reconstructed = results[k]['predictions'][sample_idx].reshape(28, 28)
                
                # Show original and reconstruction
                ax = axes[k_idx, sample_idx]
                comparison = np.hstack([original, reconstructed])
                ax.imshow(comparison, cmap='gray')
                ax.axis('off')
                
                if sample_idx == 0:
                    ax.set_ylabel(f'k={k}\nMSE={results[k]["mse"]:.4f}', 
                                fontsize=10, fontweight='bold')
                if k_idx == 0:
                    ax.set_title(f'Sample {sample_idx+1}', fontsize=10)
    
    plt.suptitle('Detailed Reconstruction Comparison: Original | Reconstructed\n(Improved Algorithm with JumpReLU + Comprehensive Loss)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/detailed_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Results Generated Successfully!")
    print("=" * 60)
    print("📁 Generated Files:")
    print("  • images/comprehensive_k_sparse_results.png")
    print("  • images/detailed_reconstruction_comparison.png")
    
    print("\n📊 Results Summary:")
    for k in k_values:
        print(f"  k={k:2d}: MSE={results[k]['mse']:.4f}, Loss={results[k]['final_loss']:.4f}, Time={results[k]['training_time']:.1f}s")
    
    return results

if __name__ == "__main__":
    results = main()