#!/usr/bin/env python3
"""
Generate results with basic working algorithm including k=50
"""
import numpy as np
import matplotlib.pyplot as plt
from utilis.mnist.mnist_helper import MnistHelper
from layers.sparse_layer import SparseLayer
from layers.linear_layer import LinearLayer
from nets.fcnn import FCNeuralNet
from utilis.activations import sigmoid_function
import time
import os

def create_basic_model(k_sparse=25):
    """Create basic working model."""
    # Create layers
    encoder = SparseLayer("encoder", 784, 100, sigmoid_function, k_sparse)
    decoder = LinearLayer("decoder", 100, 784, sigmoid_function)
    
    # Create network
    model = FCNeuralNet([encoder, decoder])
    
    return model

def main():
    print("🚀 Generating Results with Basic Working Algorithm")
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
        
        # Create basic model
        model = create_basic_model(k_sparse=k)
        
        # Train model
        start_time = time.time()
        model.train(
            train_data, train_data,
            epochs=50,
            learning_rate=0.1,
            batch_size=64,
            print_epochs=25
        )
        training_time = time.time() - start_time
        
        # Evaluate on test data
        predictions = model.predict(test_data)
        mse = np.mean((test_data - predictions) ** 2)
        
        results[k] = {
            'mse': mse,
            'training_time': training_time,
            'predictions': predictions[:20]  # Save first 20 for visualization
        }
        
        print(f"  ✅ k={k}: MSE={mse:.4f}, Time={training_time:.1f}s")
    
    # Create visualization
    print("\n📊 Creating comprehensive visualization...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance comparison
    ax1 = plt.subplot(3, 4, 1)
    k_list = list(results.keys())
    mse_list = [results[k]['mse'] for k in k_list]
    
    plt.plot(k_list, mse_list, 'bo-', linewidth=2, markersize=8)
    plt.title('Reconstruction Quality vs Sparsity\n(K-Sparse AutoEncoder)', fontsize=12, fontweight='bold')
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
    
    # 3. Sparsity vs Compression
    ax3 = plt.subplot(3, 4, 3)
    compression_ratios = [1 - (k/100) for k in k_list]
    plt.plot(k_list, compression_ratios, 'ro-', linewidth=2, markersize=8)
    plt.title('Compression Ratio vs k', fontsize=12, fontweight='bold')
    plt.xlabel('k (Active Neurons)')
    plt.ylabel('Compression Ratio')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    
    # 4. Quality vs Sparsity tradeoff
    ax4 = plt.subplot(3, 4, 4)
    sparsity_levels = [k/100 for k in k_list]
    plt.scatter(sparsity_levels, mse_list, s=100, alpha=0.7, c='purple')
    plt.title('Quality vs Sparsity Tradeoff', fontsize=12, fontweight='bold')
    plt.xlabel('Sparsity Level (k/100)')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    
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
    
    # 9-12. Show multiple samples for different k values
    for i, k in enumerate([5, 10, 30, 50]):
        ax = plt.subplot(3, 4, 9 + i)
        sample_idx = i + 1
        original = test_data[sample_idx].reshape(28, 28)
        reconstructed = results[k]['predictions'][sample_idx].reshape(28, 28)
        
        # Create side-by-side comparison
        comparison = np.hstack([original, reconstructed])
        plt.imshow(comparison, cmap='gray')
        plt.title(f'k={k} Sample {sample_idx+1}', fontsize=10)
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
    
    plt.suptitle('Detailed Reconstruction Comparison: Original | Reconstructed\n(K-Sparse AutoEncoder with Different Sparsity Levels)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/detailed_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate figure just for k=50 analysis
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(10):
        row = i // 5
        col = i % 5
        
        original = test_data[i].reshape(28, 28)
        reconstructed = results[50]['predictions'][i].reshape(28, 28)
        
        # Show original and reconstruction
        ax = axes[row, col]
        comparison = np.hstack([original, reconstructed])
        ax.imshow(comparison, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Sample {i+1}', fontsize=10)
    
    plt.suptitle(f'k=50 Reconstruction Analysis (MSE={results[50]["mse"]:.4f})\nOriginal | Reconstructed', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/k50_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Results Generated Successfully!")
    print("=" * 60)
    print("📁 Generated Files:")
    print("  • images/comprehensive_k_sparse_results.png")
    print("  • images/detailed_reconstruction_comparison.png")
    print("  • images/k50_analysis.png")
    
    print("\n📊 Results Summary:")
    for k in k_values:
        print(f"  k={k:2d}: MSE={results[k]['mse']:.4f}, Time={results[k]['training_time']:.1f}s")
    
    return results

if __name__ == "__main__":
    results = main()