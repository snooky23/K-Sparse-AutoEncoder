#!/usr/bin/env python3
"""
Generate high-quality figures with properly trained models
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utilis.mnist.mnist_helper import MnistHelper
from layers.sparse_layer import SparseLayer
from layers.linear_layer import LinearLayer
from nets.fcnn import FCNeuralNet
from utilis.activations import sigmoid_function
import time
import os

# Set professional plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')

sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif'
})

def create_model(k_sparse=25):
    """Create K-Sparse AutoEncoder model."""
    encoder = SparseLayer("encoder", 784, 100, sigmoid_function, k_sparse)
    decoder = LinearLayer("decoder", 100, 784, sigmoid_function)
    return FCNeuralNet([encoder, decoder])

def create_quality_analysis():
    """Create high-quality analysis with properly trained models."""
    # Load data
    print("📊 Loading MNIST data...")
    mnist = MnistHelper()
    train_lbl, train_img, test_lbl, test_img = mnist.get_data()
    
    # Use larger dataset for better training
    train_data = train_img.reshape(-1, 784)[:10000] / 255.0
    test_data = test_img.reshape(-1, 784)[:1000] / 255.0
    
    k_values = [5, 10, 20, 30, 50]
    results = {}
    
    print("🧠 Training models with proper epochs...")
    
    for k in k_values:
        print(f"\n🔹 Training k={k} with 100 epochs...")
        model = create_model(k_sparse=k)
        
        # Proper training with more epochs
        start_time = time.time()
        model.train(train_data, train_data, epochs=100, learning_rate=0.1, batch_size=64, print_epochs=50)
        training_time = time.time() - start_time
        
        # Evaluate
        predictions = model.predict(test_data)
        mse = np.mean((test_data - predictions) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        results[k] = {
            'mse': mse,
            'psnr': psnr,
            'training_time': training_time,
            'sparsity_ratio': k / 100,
            'compression_ratio': 1 - (k / 100),
            'predictions': predictions[:20]  # Store more samples
        }
        
        print(f"  ✅ k={k}: MSE={mse:.4f}, PSNR={psnr:.1f}dB, Time={training_time:.1f}s")
    
    return results, test_data

def create_reconstruction_showcase():
    """Create high-quality reconstruction showcase."""
    results, test_data = create_quality_analysis()
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Quality metrics comparison
    ax1 = plt.subplot(3, 4, 1)
    k_list = list(results.keys())
    mse_list = [results[k]['mse'] for k in k_list]
    psnr_list = [results[k]['psnr'] for k in k_list]
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(k_list, mse_list, 'o-', color='red', linewidth=3, markersize=8, label='MSE')
    line2 = ax1_twin.plot(k_list, psnr_list, 's-', color='blue', linewidth=3, markersize=8, label='PSNR')
    
    ax1.set_xlabel('k (Active Neurons)')
    ax1.set_ylabel('MSE', color='red')
    ax1_twin.set_ylabel('PSNR (dB)', color='blue')
    ax1.set_title('Quality vs Sparsity Trade-off', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # 2. Compression efficiency
    ax2 = plt.subplot(3, 4, 2)
    compression_ratios = [results[k]['compression_ratio'] for k in k_list]
    
    bars = ax2.bar(k_list, compression_ratios, alpha=0.8, color='green', edgecolor='black')
    ax2.set_xlabel('k (Active Neurons)')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression Efficiency', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, ratio in zip(bars, compression_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Training time analysis
    ax3 = plt.subplot(3, 4, 3)
    training_times = [results[k]['training_time'] for k in k_list]
    
    ax3.plot(k_list, training_times, 'o-', color='purple', linewidth=3, markersize=8)
    ax3.set_xlabel('k (Active Neurons)')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('Training Efficiency', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Quality-Compression Pareto frontier
    ax4 = plt.subplot(3, 4, 4)
    scatter = ax4.scatter(compression_ratios, mse_list, c=k_list, s=150, 
                         alpha=0.8, cmap='viridis', edgecolors='black', linewidth=2)
    ax4.set_xlabel('Compression Ratio')
    ax4.set_ylabel('MSE')
    ax4.set_title('Quality-Compression Pareto Frontier', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add k value labels
    for i, k in enumerate(k_list):
        ax4.annotate(f'k={k}', (compression_ratios[i], mse_list[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # 5-12. High-quality reconstruction examples
    for i, k in enumerate(k_list):
        # Original vs Reconstructed comparison
        ax = plt.subplot(3, 4, 5 + i)
        
        # Select a clear digit example
        digit_idx = i  # Use different digits for each k
        original = test_data[digit_idx].reshape(28, 28)
        reconstructed = results[k]['predictions'][digit_idx].reshape(28, 28)
        
        # Create side-by-side comparison
        comparison = np.hstack([original, reconstructed])
        
        im = ax.imshow(comparison, cmap='gray', interpolation='nearest')
        ax.set_title(f'k={k} | MSE={results[k]["mse"]:.4f}', fontweight='bold', fontsize=12)
        ax.axis('off')
        
        # Add separator line
        ax.axvline(x=27.5, color='red', linewidth=2, alpha=0.7)
        
        # Add labels
        ax.text(14, -2, 'Original', ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(42, -2, 'Reconstructed', ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.suptitle('High-Quality K-Sparse AutoEncoder Analysis\n(Properly Trained Models)', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/high_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed reconstruction grid
    create_detailed_reconstruction_grid(results, test_data)
    
    return results

def create_detailed_reconstruction_grid(results, test_data):
    """Create detailed reconstruction grid showing multiple examples."""
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    
    k_values = list(results.keys())
    
    for k_idx, k in enumerate(k_values):
        for sample_idx in range(10):
            ax = axes[k_idx, sample_idx]
            
            # Get original and reconstructed
            original = test_data[sample_idx].reshape(28, 28)
            reconstructed = results[k]['predictions'][sample_idx].reshape(28, 28)
            
            # Create side-by-side comparison
            comparison = np.hstack([original, reconstructed])
            
            ax.imshow(comparison, cmap='gray', interpolation='nearest')
            ax.axis('off')
            
            # Add separator line
            ax.axvline(x=27.5, color='red', linewidth=1, alpha=0.5)
            
            # Add k value label on first column
            if sample_idx == 0:
                ax.text(-5, 14, f'k={k}\nMSE={results[k]["mse"]:.4f}', 
                       ha='right', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            # Add sample number on first row
            if k_idx == 0:
                ax.text(28, -3, f'Sample {sample_idx+1}', ha='center', va='top', 
                       fontsize=10, fontweight='bold')
    
    plt.suptitle('Detailed Reconstruction Comparison: Original | Reconstructed\n(High-Quality Training Results)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/detailed_high_quality_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_architecture_diagram():
    """Create clean architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # Define components
    components = [
        {'name': 'Input\n(28×28 MNIST)', 'pos': (1, 2), 'size': (1.5, 2), 'color': '#E3F2FD'},
        {'name': 'Encoder\n(784→100)', 'pos': (4, 2), 'size': (1.5, 2), 'color': '#FFEBEE'},
        {'name': 'Top-k\nSelection\n(k active)', 'pos': (7, 2), 'size': (1.5, 2), 'color': '#FFF3E0'},
        {'name': 'Decoder\n(100→784)', 'pos': (10, 2), 'size': (1.5, 2), 'color': '#E8F5E8'},
        {'name': 'Reconstruction\n(28×28)', 'pos': (13, 2), 'size': (1.5, 2), 'color': '#F3E5F5'}
    ]
    
    # Draw components
    for comp in components:
        rect = plt.Rectangle(comp['pos'], comp['size'][0], comp['size'][1], 
                           facecolor=comp['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2,
                comp['name'], ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=3, color='black')
    ax.annotate('', xy=(4, 3), xytext=(2.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 3), xytext=(5.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 3), xytext=(8.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(13, 3), xytext=(11.5, 3), arrowprops=arrow_props)
    
    # Add gradient flow annotation
    ax.text(7.75, 0.5, 'Differentiable\nGradient Flow', ha='center', va='center',
            fontsize=11, style='italic', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('K-Sparse AutoEncoder: Differentiable Architecture', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('images/clean_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate high-quality figures with properly trained models."""
    print("🎨 Generating high-quality figures with proper training...")
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Generate architecture diagram
    print("1. Creating clean architecture diagram...")
    create_architecture_diagram()
    
    # Generate high-quality analysis
    print("2. Creating high-quality reconstruction analysis...")
    results = create_reconstruction_showcase()
    
    print("\n✅ High-quality figures generated successfully!")
    print("📁 Generated files:")
    print("  • images/clean_architecture.png")
    print("  • images/high_quality_analysis.png")
    print("  • images/detailed_high_quality_reconstructions.png")
    
    print("\n📊 Results Summary:")
    for k, data in results.items():
        print(f"  k={k:2d}: MSE={data['mse']:.4f}, PSNR={data['psnr']:.1f}dB, Time={data['training_time']:.1f}s")
    
    return results

if __name__ == "__main__":
    main()