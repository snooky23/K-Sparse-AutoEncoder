"""Enhanced visualization tools for K-Sparse AutoEncoder.

This module provides comprehensive visualization capabilities including:
- Training progress visualization
- Model architecture diagrams
- Weight and activation visualizations
- Sparsity analysis plots
- Comparison visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import os
from datetime import datetime

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualizer for training progress and metrics."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize training visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_training_history(self, history: Dict[str, List], 
                            save_path: Optional[str] = None,
                            show_sparsity: bool = True) -> None:
        """Plot comprehensive training history.
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save plot
            show_sparsity: Whether to show sparsity information
        """
        # Determine subplot layout
        n_plots = 2 if not show_sparsity else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot loss
        if 'loss' in history:
            axes[0].plot(history['loss'], color=self.colors[0], linewidth=2)
            axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
            
            # Add trend line
            if len(history['loss']) > 10:
                x = np.arange(len(history['loss']))
                z = np.polyfit(x, history['loss'], 1)
                p = np.poly1d(z)
                axes[0].plot(x, p(x), "--", color=self.colors[1], alpha=0.7, 
                           label=f'Trend: {z[0]:.2e}')
                axes[0].legend()
        
        # Plot learning rate
        if 'learning_rate' in history:
            axes[1].plot(history['learning_rate'], color=self.colors[2], linewidth=2)
            axes[1].set_title('Learning Rate', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        
        # Plot sparsity information
        if show_sparsity and 'sparsity_info' in history and len(axes) > 2:
            sparsity_data = history['sparsity_info']
            if sparsity_data:
                # Extract sparsity metrics
                epochs = range(len(sparsity_data))
                for layer_name in sparsity_data[0].keys():
                    actual_k_means = [info[layer_name].get('actual_k_mean', 0) 
                                    for info in sparsity_data]
                    axes[2].plot(epochs, actual_k_means, 
                               label=f'{layer_name} (k)', linewidth=2)
                
                axes[2].set_title('Sparsity Level', fontsize=12, fontweight='bold')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Active Neurons')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_loss_comparison(self, histories: Dict[str, Dict[str, List]], 
                           save_path: Optional[str] = None) -> None:
        """Plot loss comparison between different experiments.
        
        Args:
            histories: Dictionary of experiment names to histories
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        for i, (name, history) in enumerate(histories.items()):
            if 'loss' in history:
                plt.plot(history['loss'], 
                        color=self.colors[i % len(self.colors)], 
                        linewidth=2, label=name)
        
        plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence_analysis(self, history: Dict[str, List], 
                                window_size: int = 10,
                                save_path: Optional[str] = None) -> None:
        """Plot convergence analysis with moving averages.
        
        Args:
            history: Training history
            window_size: Window size for moving average
            save_path: Optional path to save plot
        """
        if 'loss' not in history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        loss = np.array(history['loss'])
        epochs = np.arange(len(loss))
        
        # Plot raw loss and moving average
        ax1.plot(epochs, loss, alpha=0.3, color=self.colors[0], label='Raw Loss')
        
        # Calculate moving average
        if len(loss) >= window_size:
            moving_avg = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(epochs[window_size-1:], moving_avg, 
                    color=self.colors[1], linewidth=2, label=f'Moving Avg ({window_size})')
        
        ax1.set_title('Loss Convergence', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss improvement rate
        if len(loss) > 1:
            improvement = np.diff(loss)
            ax2.plot(epochs[1:], improvement, color=self.colors[2], linewidth=1)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.set_title('Loss Improvement Rate', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Change')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ModelVisualizer:
    """Visualizer for model architecture and weights."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize model visualizer."""
        self.figsize = figsize
    
    def plot_architecture(self, layers: List[Any], 
                         save_path: Optional[str] = None) -> None:
        """Plot model architecture diagram.
        
        Args:
            layers: List of network layers
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate positions
        layer_positions = []
        max_neurons = max(layer.n_out for layer in layers)
        layer_width = 0.8 / len(layers)
        
        for i, layer in enumerate(layers):
            x = 0.1 + i * layer_width * 1.2
            layer_positions.append((x, layer))
        
        # Draw layers
        for i, (x, layer) in enumerate(layer_positions):
            # Draw layer box
            height = 0.6 * (layer.n_out / max_neurons)
            y = 0.5 - height / 2
            
            # Choose color based on layer type
            if hasattr(layer, 'num_k_sparse'):
                color = 'lightcoral'
                label = f'Sparse Layer\\n{layer.name}\\n({layer.n_in}→{layer.n_out})\\nk={layer.num_k_sparse}'
            else:
                color = 'lightblue'
                label = f'Linear Layer\\n{layer.name}\\n({layer.n_in}→{layer.n_out})'
            
            # Draw rectangle
            rect = FancyBboxPatch((x - layer_width/2, y), layer_width, height,
                                boxstyle="round,pad=0.01", 
                                facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x, 0.5, label, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            
            # Draw connections to next layer
            if i < len(layer_positions) - 1:
                next_x = layer_positions[i + 1][0]
                arrow = patches.FancyArrowPatch((x + layer_width/2, 0.5),
                                             (next_x - layer_width/2, 0.5),
                                             arrowstyle='->', mutation_scale=20,
                                             color='gray', alpha=0.7)
                ax.add_patch(arrow)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Model Architecture', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_weight_distribution(self, layers: List[Any], 
                               save_path: Optional[str] = None) -> None:
        """Plot weight distribution for all layers.
        
        Args:
            layers: List of network layers
            save_path: Optional path to save plot
        """
        n_layers = len(layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, layer in enumerate(layers):
            weights = layer.weights.flatten()
            
            # Plot histogram
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            axes[i].hist(weights, bins=50, alpha=0.7, color=colors[i % len(colors)])
            axes[i].set_title(f'{layer.name} Weights', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)
            axes[i].axvline(mean_weight, color='red', linestyle='--', 
                          label=f'Mean: {mean_weight:.3f}')
            axes[i].axvline(mean_weight + std_weight, color='red', linestyle=':', alpha=0.5)
            axes[i].axvline(mean_weight - std_weight, color='red', linestyle=':', alpha=0.5)
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_weight_heatmap(self, layer: Any, 
                           save_path: Optional[str] = None) -> None:
        """Plot weight matrix as heatmap.
        
        Args:
            layer: Network layer
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot weight matrix
        sns.heatmap(layer.weights, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Weight Value'})
        
        plt.title(f'{layer.name} Weight Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Output Neurons')
        plt.ylabel('Input Neurons')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class SparsityVisualizer:
    """Visualizer for sparsity analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize sparsity visualizer."""
        self.figsize = figsize
    
    def plot_sparsity_patterns(self, activations: np.ndarray, 
                             k_values: List[int],
                             save_path: Optional[str] = None) -> None:
        """Plot sparsity patterns for different k values.
        
        Args:
            activations: Activation matrix (samples x neurons)
            k_values: List of k values to analyze
            save_path: Optional path to save plot
        """
        n_k = len(k_values)
        fig, axes = plt.subplots(1, n_k, figsize=(4 * n_k, 4))
        
        if n_k == 1:
            axes = [axes]
        
        for i, k in enumerate(k_values):
            # Create sparse activations
            sparse_activations = np.zeros_like(activations)
            for j in range(activations.shape[0]):
                indices = np.argpartition(activations[j], -k)[-k:]
                sparse_activations[j, indices] = activations[j, indices]
            
            # Plot sparsity pattern
            im = axes[i].imshow(sparse_activations.T, cmap='viridis', aspect='auto')
            axes[i].set_title(f'Sparsity Pattern (k={k})', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Samples')
            axes[i].set_ylabel('Neurons')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Activation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sparsity_statistics(self, sparse_layer: Any, 
                                save_path: Optional[str] = None) -> None:
        """Plot sparsity statistics for a sparse layer.
        
        Args:
            sparse_layer: Sparse layer with sparsity information
            save_path: Optional path to save plot
        """
        if not hasattr(sparse_layer, 'sparsity_mask') or sparse_layer.sparsity_mask is None:
            print("No sparsity information available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Active neurons per sample
        active_per_sample = np.sum(sparse_layer.sparsity_mask, axis=1)
        axes[0, 0].hist(active_per_sample, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Active Neurons per Sample', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Active Neurons')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(sparse_layer.num_k_sparse, color='red', linestyle='--', 
                          label=f'Target k={sparse_layer.num_k_sparse}')
        axes[0, 0].legend()
        
        # Neuron activation frequency
        activation_freq = np.sum(sparse_layer.sparsity_mask, axis=0) / sparse_layer.sparsity_mask.shape[0]
        axes[0, 1].bar(range(len(activation_freq)), activation_freq, alpha=0.7, color='green')
        axes[0, 1].set_title('Neuron Activation Frequency', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Neuron Index')
        axes[0, 1].set_ylabel('Activation Frequency')
        axes[0, 1].axhline(sparse_layer.num_k_sparse / sparse_layer.sparsity_mask.shape[1], 
                          color='red', linestyle='--', label='Expected Frequency')
        axes[0, 1].legend()
        
        # Sparsity pattern
        axes[1, 0].imshow(sparse_layer.sparsity_mask.T, cmap='binary', aspect='auto')
        axes[1, 0].set_title('Sparsity Pattern', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Samples')
        axes[1, 0].set_ylabel('Neurons')
        
        # Activation distribution
        if hasattr(sparse_layer, 'sparse_activations') and sparse_layer.sparse_activations is not None:
            non_zero_activations = sparse_layer.sparse_activations[sparse_layer.sparse_activations > 0]
            axes[1, 1].hist(non_zero_activations, bins=50, alpha=0.7, color='orange')
            axes[1, 1].set_title('Non-zero Activation Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Activation Value')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ComparisonVisualizer:
    """Visualizer for comparing different models and experiments."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize comparison visualizer."""
        self.figsize = figsize
    
    def plot_performance_comparison(self, results: Dict[str, Dict[str, float]], 
                                  metrics: List[str] = ['mse', 'loss'],
                                  save_path: Optional[str] = None) -> None:
        """Plot performance comparison between different experiments.
        
        Args:
            results: Dictionary of experiment names to results
            metrics: List of metrics to compare
            save_path: Optional path to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            experiments = list(results.keys())
            values = [results[exp].get(metric, 0) for exp in experiments]
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            bars = axes[i].bar(experiments, values, alpha=0.7, 
                             color=[colors[j % len(colors)] for j in range(len(experiments))])
            axes[i].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_k_value_analysis(self, k_results: Dict[int, Dict[str, float]], 
                            save_path: Optional[str] = None) -> None:
        """Plot analysis of different k values.
        
        Args:
            k_results: Dictionary of k values to results
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        k_values = sorted(k_results.keys())
        mse_values = [k_results[k].get('mse', 0) for k in k_values]
        loss_values = [k_results[k].get('loss', 0) for k in k_values]
        
        # Plot MSE vs k
        ax1.plot(k_values, mse_values, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_title('Reconstruction Quality vs Sparsity', fontsize=12, fontweight='bold')
        ax1.set_xlabel('k (Number of Active Neurons)')
        ax1.set_ylabel('MSE')
        ax1.grid(True, alpha=0.3)
        
        # Plot loss vs k
        ax2.plot(k_values, loss_values, 'o-', linewidth=2, markersize=8, color='red')
        ax2.set_title('Training Loss vs Sparsity', fontsize=12, fontweight='bold')
        ax2.set_xlabel('k (Number of Active Neurons)')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ReconstructionVisualizer:
    """Visualizer for reconstruction results."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """Initialize reconstruction visualizer."""
        self.figsize = figsize
    
    def plot_reconstruction_comparison(self, original: np.ndarray, 
                                     reconstructions: Dict[str, np.ndarray],
                                     n_samples: int = 5,
                                     save_path: Optional[str] = None) -> None:
        """Plot reconstruction comparison for different methods.
        
        Args:
            original: Original images
            reconstructions: Dictionary of method names to reconstructions
            n_samples: Number of samples to show
            save_path: Optional path to save plot
        """
        n_methods = len(reconstructions) + 1  # +1 for original
        fig, axes = plt.subplots(n_methods, n_samples, figsize=self.figsize)
        
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        # Show original images
        for i in range(n_samples):
            img = original[i].reshape(28, 28)
            axes[0, i].imshow(img, cmap='gray')
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=12, fontweight='bold')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
        
        # Show reconstructions
        for method_idx, (method_name, reconstruction) in enumerate(reconstructions.items()):
            for i in range(n_samples):
                img = reconstruction[i].reshape(28, 28)
                axes[method_idx + 1, i].imshow(img, cmap='gray')
                if i == 0:
                    axes[method_idx + 1, i].set_ylabel(method_name, fontsize=12, fontweight='bold')
                axes[method_idx + 1, i].set_xticks([])
                axes[method_idx + 1, i].set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Convenience functions
def create_visualization_suite(output_dir: str = "visualizations/") -> Dict[str, Any]:
    """Create a complete visualization suite.
    
    Args:
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of visualizer instances
    """
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'training': TrainingVisualizer(),
        'model': ModelVisualizer(),
        'sparsity': SparsityVisualizer(),
        'comparison': ComparisonVisualizer(),
        'reconstruction': ReconstructionVisualizer(),
        'output_dir': output_dir
    }


def save_experiment_report(experiment_name: str, 
                          results: Dict[str, Any],
                          visualizations: Dict[str, Any],
                          output_dir: str = "reports/") -> str:
    """Save comprehensive experiment report.
    
    Args:
        experiment_name: Name of the experiment
        results: Experiment results
        visualizations: Generated visualizations
        output_dir: Directory to save report
        
    Returns:
        Path to saved report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f"{experiment_name}_report.html")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{experiment_name} - Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{experiment_name}</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Results Summary</h2>
            {_format_results_html(results)}
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            {_format_visualizations_html(visualizations)}
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path


def _format_results_html(results: Dict[str, Any]) -> str:
    """Format results as HTML."""
    html = ""
    for key, value in results.items():
        if isinstance(value, (int, float)):
            html += f'<div class="metric"><strong>{key}:</strong> {value:.4f}</div>'
        else:
            html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
    return html


def _format_visualizations_html(visualizations: Dict[str, Any]) -> str:
    """Format visualizations as HTML."""
    html = ""
    for viz_name, viz_path in visualizations.items():
        if viz_path and os.path.exists(viz_path):
            html += f'<div class="visualization"><h3>{viz_name}</h3><img src="{viz_path}" style="max-width: 100%;"></div>'
    return html