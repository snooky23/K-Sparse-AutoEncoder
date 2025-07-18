"""Comprehensive benchmarking suite for K-Sparse AutoEncoder.

This module provides extensive benchmarking capabilities including:
- Performance benchmarking (speed, memory usage)
- Model quality evaluation
- Scalability analysis
- Comparison with baseline methods
- Statistical significance testing
"""
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy import stats
import json
import os
from datetime import datetime

from utilis.config import ExperimentConfig
from utilis.visualization import create_visualization_suite


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            metrics=data['metrics'],
            metadata=data['metadata'],
            timestamp=data['timestamp']
        )


class PerformanceBenchmark:
    """Benchmark for performance metrics (speed, memory)."""
    
    def __init__(self):
        """Initialize performance benchmark."""
        self.results = []
    
    def benchmark_training_speed(self, 
                                train_function: Callable,
                                data: Tuple[np.ndarray, np.ndarray],
                                config: ExperimentConfig,
                                n_runs: int = 3) -> BenchmarkResult:
        """Benchmark training speed.
        
        Args:
            train_function: Function to benchmark
            data: Training data (x, y)
            config: Configuration
            n_runs: Number of runs for averaging
            
        Returns:
            Benchmark result
        """
        times = []
        memory_usage = []
        
        for run in range(n_runs):
            # Monitor memory before training
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time training
            start_time = time.time()
            result = train_function(data[0], data[1], config)
            end_time = time.time()
            
            # Monitor memory after training
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
        
        metrics = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage),
            'throughput': len(data[0]) / np.mean(times)  # samples per second
        }
        
        metadata = {
            'n_runs': n_runs,
            'data_size': len(data[0]),
            'config': config.name if hasattr(config, 'name') else 'unnamed'
        }
        
        result = BenchmarkResult(
            name=f"training_speed_{config.name if hasattr(config, 'name') else 'unnamed'}",
            metrics=metrics,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def benchmark_inference_speed(self,
                                 model: Any,
                                 data: np.ndarray,
                                 batch_sizes: List[int] = [1, 16, 32, 64, 128],
                                 n_runs: int = 10) -> List[BenchmarkResult]:
        """Benchmark inference speed for different batch sizes.
        
        Args:
            model: Model to benchmark
            data: Test data
            batch_sizes: List of batch sizes to test
            n_runs: Number of runs per batch size
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for batch_size in batch_sizes:
            times = []
            
            for run in range(n_runs):
                # Select random batch
                if batch_size <= len(data):
                    batch_indices = np.random.choice(len(data), batch_size, replace=False)
                    batch_data = data[batch_indices]
                else:
                    batch_data = data
                
                # Time inference
                start_time = time.time()
                predictions = model.predict(batch_data)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            metrics = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'throughput': batch_size / np.mean(times),
                'time_per_sample': np.mean(times) / batch_size
            }
            
            metadata = {
                'batch_size': batch_size,
                'n_runs': n_runs,
                'data_size': len(data)
            }
            
            result = BenchmarkResult(
                name=f"inference_speed_batch_{batch_size}",
                metrics=metrics,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_memory_usage(self,
                              model: Any,
                              data_sizes: List[int] = [100, 500, 1000, 2000, 5000],
                              generate_data_fn: Optional[Callable] = None) -> List[BenchmarkResult]:
        """Benchmark memory usage for different data sizes.
        
        Args:
            model: Model to benchmark
            data_sizes: List of data sizes to test
            generate_data_fn: Function to generate data of specified size
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for data_size in data_sizes:
            if generate_data_fn:
                data = generate_data_fn(data_size)
            else:
                data = np.random.randn(data_size, 784)
            
            # Monitor memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Forward pass
            predictions = model.predict(data)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            metrics = {
                'memory_used_mb': memory_used,
                'memory_per_sample': memory_used / data_size,
                'peak_memory_mb': memory_after
            }
            
            metadata = {
                'data_size': data_size,
                'data_shape': data.shape
            }
            
            result = BenchmarkResult(
                name=f"memory_usage_{data_size}",
                metrics=metrics,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
        
        self.results.extend(results)
        return results


class QualityBenchmark:
    """Benchmark for model quality metrics."""
    
    def __init__(self):
        """Initialize quality benchmark."""
        self.results = []
    
    def benchmark_reconstruction_quality(self,
                                       models: Dict[str, Any],
                                       test_data: np.ndarray,
                                       metrics: List[str] = ['mse', 'mae', 'ssim']) -> List[BenchmarkResult]:
        """Benchmark reconstruction quality across different models.
        
        Args:
            models: Dictionary of model names to models
            test_data: Test data
            metrics: List of metrics to compute
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for model_name, model in models.items():
            # Get reconstructions
            reconstructions = model.predict(test_data)
            
            # Calculate metrics
            computed_metrics = {}
            
            if 'mse' in metrics:
                computed_metrics['mse'] = np.mean((test_data - reconstructions) ** 2)
            
            if 'mae' in metrics:
                computed_metrics['mae'] = np.mean(np.abs(test_data - reconstructions))
            
            if 'rmse' in metrics:
                computed_metrics['rmse'] = np.sqrt(np.mean((test_data - reconstructions) ** 2))
            
            if 'ssim' in metrics:
                # Simplified SSIM calculation
                computed_metrics['ssim'] = self._calculate_ssim(test_data, reconstructions)
            
            # Add sparsity metrics if applicable
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    if hasattr(layer, 'num_k_sparse'):
                        computed_metrics['target_sparsity'] = layer.num_k_sparse
                        if hasattr(layer, 'sparsity_mask') and layer.sparsity_mask is not None:
                            computed_metrics['actual_sparsity'] = np.mean(np.sum(layer.sparsity_mask, axis=1))
            
            metadata = {
                'model_name': model_name,
                'test_size': len(test_data),
                'data_shape': test_data.shape
            }
            
            result = BenchmarkResult(
                name=f"quality_{model_name}",
                metrics=computed_metrics,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_sparsity_effectiveness(self,
                                       model: Any,
                                       test_data: np.ndarray,
                                       k_values: List[int]) -> List[BenchmarkResult]:
        """Benchmark sparsity effectiveness across different k values.
        
        Args:
            model: Model to benchmark
            test_data: Test data
            k_values: List of k values to test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for k in k_values:
            # Temporarily modify k value
            original_k = None
            for layer in model.layers:
                if hasattr(layer, 'num_k_sparse'):
                    original_k = layer.num_k_sparse
                    layer.num_k_sparse = k
                    if hasattr(layer, 'sparse_activation'):
                        layer.sparse_activation.num_k_sparse = k
            
            # Get reconstructions
            reconstructions = model.predict(test_data)
            
            # Calculate metrics
            mse = np.mean((test_data - reconstructions) ** 2)
            sparsity_ratio = k / model.layers[0].n_out if hasattr(model, 'layers') else 0
            
            metrics = {
                'mse': mse,
                'k_value': k,
                'sparsity_ratio': sparsity_ratio,
                'compression_ratio': 1 - sparsity_ratio
            }
            
            metadata = {
                'original_k': original_k,
                'test_size': len(test_data)
            }
            
            result = BenchmarkResult(
                name=f"sparsity_k_{k}",
                metrics=metrics,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
            
            # Restore original k value
            if original_k is not None:
                for layer in model.layers:
                    if hasattr(layer, 'num_k_sparse'):
                        layer.num_k_sparse = original_k
                        if hasattr(layer, 'sparse_activation'):
                            layer.sparse_activation.num_k_sparse = original_k
        
        self.results.extend(results)
        return results
    
    def _calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate simplified SSIM metric."""
        # Simplified SSIM implementation
        mu1 = np.mean(original)
        mu2 = np.mean(reconstructed)
        
        sigma1_sq = np.var(original)
        sigma2_sq = np.var(reconstructed)
        sigma12 = np.mean((original - mu1) * (reconstructed - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim


class ScalabilityBenchmark:
    """Benchmark for scalability analysis."""
    
    def __init__(self):
        """Initialize scalability benchmark."""
        self.results = []
    
    def benchmark_scaling_performance(self,
                                    create_model_fn: Callable,
                                    train_function: Callable,
                                    hidden_sizes: List[int] = [50, 100, 200, 400],
                                    data_sizes: List[int] = [1000, 2000, 5000, 10000],
                                    generate_data_fn: Optional[Callable] = None) -> List[BenchmarkResult]:
        """Benchmark performance scaling with model and data size.
        
        Args:
            create_model_fn: Function to create model with given hidden size
            train_function: Function to train model
            hidden_sizes: List of hidden layer sizes
            data_sizes: List of data sizes
            generate_data_fn: Function to generate data
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for hidden_size in hidden_sizes:
            for data_size in data_sizes:
                # Create model
                model = create_model_fn(hidden_size)
                
                # Generate data
                if generate_data_fn:
                    data = generate_data_fn(data_size)
                else:
                    data = (np.random.randn(data_size, 784), np.random.randn(data_size, 784))
                
                # Benchmark training
                start_time = time.time()
                history = train_function(model, data[0], data[1], epochs=10)
                end_time = time.time()
                
                # Calculate metrics
                training_time = end_time - start_time
                final_loss = history['loss'][-1] if 'loss' in history else 0
                
                # Calculate model parameters
                total_params = sum(layer.weights.size + layer.biases.size 
                                 for layer in model.layers if hasattr(layer, 'weights'))
                
                metrics = {
                    'training_time': training_time,
                    'final_loss': final_loss,
                    'total_parameters': total_params,
                    'time_per_parameter': training_time / total_params,
                    'time_per_sample': training_time / data_size
                }
                
                metadata = {
                    'hidden_size': hidden_size,
                    'data_size': data_size,
                    'epochs': 10
                }
                
                result = BenchmarkResult(
                    name=f"scaling_h{hidden_size}_d{data_size}",
                    metrics=metrics,
                    metadata=metadata,
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
        
        self.results.extend(results)
        return results


class StatisticalAnalysis:
    """Statistical analysis of benchmark results."""
    
    @staticmethod
    def compare_methods(results1: List[BenchmarkResult],
                       results2: List[BenchmarkResult],
                       metric: str = 'mse',
                       alpha: float = 0.05) -> Dict[str, Any]:
        """Compare two sets of benchmark results statistically.
        
        Args:
            results1: First set of results
            results2: Second set of results
            metric: Metric to compare
            alpha: Significance level
            
        Returns:
            Statistical comparison results
        """
        values1 = [r.metrics.get(metric, 0) for r in results1]
        values2 = [r.metrics.get(metric, 0) for r in results2]
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(values1, values2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(values1) ** 2) + (np.std(values2) ** 2)) / 2)
        cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large',
            'mean_1': np.mean(values1),
            'mean_2': np.mean(values2),
            'std_1': np.std(values1),
            'std_2': np.std(values2)
        }
    
    @staticmethod
    def analyze_correlation(results: List[BenchmarkResult],
                          metric1: str,
                          metric2: str) -> Dict[str, Any]:
        """Analyze correlation between two metrics.
        
        Args:
            results: Benchmark results
            metric1: First metric
            metric2: Second metric
            
        Returns:
            Correlation analysis results
        """
        values1 = [r.metrics.get(metric1, 0) for r in results]
        values2 = [r.metrics.get(metric2, 0) for r in results]
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(values1, values2)
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'strength': 'weak' if abs(correlation) < 0.3 else 'moderate' if abs(correlation) < 0.7 else 'strong'
        }


class BenchmarkSuite:
    """Complete benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmarks/"):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.performance_benchmark = PerformanceBenchmark()
        self.quality_benchmark = QualityBenchmark()
        self.scalability_benchmark = ScalabilityBenchmark()
        self.statistical_analysis = StatisticalAnalysis()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_comprehensive_benchmark(self,
                                  models: Dict[str, Any],
                                  data: Dict[str, np.ndarray],
                                  configs: Dict[str, ExperimentConfig]) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.
        
        Args:
            models: Dictionary of model names to models
            data: Dictionary of data arrays
            configs: Dictionary of configurations
            
        Returns:
            Comprehensive benchmark results
        """
        results = {
            'performance': [],
            'quality': [],
            'scalability': [],
            'statistical': {}
        }
        
        print("🚀 Running Comprehensive Benchmark Suite")
        print("=" * 50)
        
        # Performance benchmarks
        print("⚡ Performance Benchmarks...")
        for model_name, model in models.items():
            # Inference speed
            inference_results = self.performance_benchmark.benchmark_inference_speed(
                model, data['test']
            )
            results['performance'].extend(inference_results)
            
            # Memory usage
            memory_results = self.performance_benchmark.benchmark_memory_usage(model)
            results['performance'].extend(memory_results)
        
        # Quality benchmarks
        print("🎯 Quality Benchmarks...")
        quality_results = self.quality_benchmark.benchmark_reconstruction_quality(
            models, data['test']
        )
        results['quality'].extend(quality_results)
        
        # Sparsity effectiveness
        for model_name, model in models.items():
            sparsity_results = self.quality_benchmark.benchmark_sparsity_effectiveness(
                model, data['test'][:100], [5, 10, 20, 30]
            )
            results['quality'].extend(sparsity_results)
        
        # Statistical analysis
        print("📊 Statistical Analysis...")
        if len(results['quality']) >= 2:
            comparison = self.statistical_analysis.compare_methods(
                results['quality'][:len(results['quality'])//2],
                results['quality'][len(results['quality'])//2:]
            )
            results['statistical']['quality_comparison'] = comparison
        
        # Save results
        self.save_results(results)
        
        # Generate report
        self.generate_report(results)
        
        print("✅ Benchmark Suite Complete!")
        return results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        results_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        
        # Convert results to serializable format
        serializable_results = {}
        for category, category_results in results.items():
            if category == 'statistical':
                serializable_results[category] = category_results
            else:
                serializable_results[category] = [
                    r.to_dict() if hasattr(r, 'to_dict') else r 
                    for r in category_results
                ]
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive benchmark report."""
        # Create visualizations
        viz_suite = create_visualization_suite(
            os.path.join(self.output_dir, "visualizations")
        )
        
        # Generate performance plots
        self._plot_performance_results(results['performance'])
        
        # Generate quality plots
        self._plot_quality_results(results['quality'])
        
        print("📈 Visualization report generated!")
    
    def _plot_performance_results(self, results: List[BenchmarkResult]) -> None:
        """Plot performance benchmark results."""
        if not results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Inference speed by batch size
        inference_results = [r for r in results if 'inference_speed' in r.name]
        if inference_results:
            batch_sizes = [r.metadata['batch_size'] for r in inference_results]
            throughputs = [r.metrics['throughput'] for r in inference_results]
            
            axes[0, 0].plot(batch_sizes, throughputs, 'o-')
            axes[0, 0].set_title('Inference Throughput vs Batch Size')
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('Throughput (samples/sec)')
            axes[0, 0].grid(True)
        
        # Memory usage by data size
        memory_results = [r for r in results if 'memory_usage' in r.name]
        if memory_results:
            data_sizes = [r.metadata['data_size'] for r in memory_results]
            memory_usage = [r.metrics['memory_used_mb'] for r in memory_results]
            
            axes[0, 1].plot(data_sizes, memory_usage, 'o-')
            axes[0, 1].set_title('Memory Usage vs Data Size')
            axes[0, 1].set_xlabel('Data Size')
            axes[0, 1].set_ylabel('Memory Usage (MB)')
            axes[0, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_benchmark.png'), dpi=300)
        plt.close()
    
    def _plot_quality_results(self, results: List[BenchmarkResult]) -> None:
        """Plot quality benchmark results."""
        if not results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # MSE comparison
        quality_results = [r for r in results if 'quality_' in r.name]
        if quality_results:
            names = [r.name.replace('quality_', '') for r in quality_results]
            mse_values = [r.metrics.get('mse', 0) for r in quality_results]
            
            axes[0, 0].bar(names, mse_values)
            axes[0, 0].set_title('MSE Comparison')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sparsity effectiveness
        sparsity_results = [r for r in results if 'sparsity_k_' in r.name]
        if sparsity_results:
            k_values = [r.metrics['k_value'] for r in sparsity_results]
            mse_values = [r.metrics['mse'] for r in sparsity_results]
            
            axes[0, 1].plot(k_values, mse_values, 'o-')
            axes[0, 1].set_title('Sparsity vs Quality')
            axes[0, 1].set_xlabel('K Value')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quality_benchmark.png'), dpi=300)
        plt.close()


# Convenience function
def run_quick_benchmark(model: Any, test_data: np.ndarray, output_dir: str = "benchmarks/") -> Dict[str, Any]:
    """Run a quick benchmark for a single model.
    
    Args:
        model: Model to benchmark
        test_data: Test data
        output_dir: Output directory
        
    Returns:
        Benchmark results
    """
    suite = BenchmarkSuite(output_dir)
    
    models = {'model': model}
    data = {'test': test_data}
    configs = {}
    
    return suite.run_comprehensive_benchmark(models, data, configs)