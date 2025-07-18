"""Performance benchmarking utilities for K-Sparse AutoEncoder.

This module provides tools for measuring and comparing performance
of different implementations and configurations.
"""
import time
import numpy as np
from typing import Dict, List, Callable, Any
from contextlib import contextmanager


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations.
    
    Args:
        name: Name of the operation being timed
        
    Yields:
        Dictionary containing timing results
    """
    start_time = time.time()
    result = {"name": name, "start_time": start_time}
    
    try:
        yield result
    finally:
        end_time = time.time()
        result["end_time"] = end_time
        result["duration"] = end_time - start_time
        print(f"{name}: {result['duration']:.4f} seconds")


class PerformanceBenchmark:
    """Performance benchmarking utilities for neural network operations."""
    
    def __init__(self):
        self.results: Dict[str, List[float]] = {}
    
    def benchmark_function(self, func: Callable, args: tuple, kwargs: dict, 
                          name: str, iterations: int = 5) -> Dict[str, float]:
        """Benchmark a function call multiple times.
        
        Args:
            func: Function to benchmark
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            name: Name for the benchmark
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        stats = {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "iterations": iterations
        }
        
        self.results[name] = times
        
        print(f"{name} - Mean: {stats['mean']:.4f}s ± {stats['std']:.4f}s "
              f"(Min: {stats['min']:.4f}s, Max: {stats['max']:.4f}s)")
        
        return stats
    
    def benchmark_layer_forward(self, layer, input_data: np.ndarray, 
                               iterations: int = 100) -> Dict[str, float]:
        """Benchmark forward pass of a layer.
        
        Args:
            layer: Layer to benchmark
            input_data: Input data for the layer
            iterations: Number of iterations
            
        Returns:
            Timing statistics
        """
        layer_name = f"{layer.__class__.__name__}_forward"
        return self.benchmark_function(
            layer.get_output, (input_data,), {}, layer_name, iterations
        )
    
    def benchmark_network_training(self, network, x_train: np.ndarray, 
                                  y_train: np.ndarray, epochs: int = 10,
                                  iterations: int = 3) -> Dict[str, float]:
        """Benchmark network training.
        
        Args:
            network: Neural network to benchmark
            x_train: Training input data
            y_train: Training target data
            epochs: Number of training epochs
            iterations: Number of benchmark iterations
            
        Returns:
            Timing statistics
        """
        # Create fresh network copies for each iteration
        import copy
        
        def train_network():
            net_copy = copy.deepcopy(network)
            net_copy.train(x_train, y_train, epochs=epochs, print_epochs=epochs+1)
        
        return self.benchmark_function(
            train_network, (), {}, "network_training", iterations
        )
    
    def memory_usage_estimate(self, data_shape: tuple, dtype: np.dtype = np.float32) -> Dict[str, float]:
        """Estimate memory usage for given data shape.
        
        Args:
            data_shape: Shape of the data
            dtype: Data type
            
        Returns:
            Memory usage estimates in MB
        """
        elements = np.prod(data_shape)
        bytes_per_element = dtype().itemsize
        total_bytes = elements * bytes_per_element
        
        return {
            "shape": data_shape,
            "elements": elements,
            "bytes": total_bytes,
            "mb": total_bytes / (1024 * 1024),
            "gb": total_bytes / (1024 * 1024 * 1024)
        }
    
    def compare_implementations(self, implementations: Dict[str, Callable],
                               args: tuple, kwargs: dict, 
                               iterations: int = 5) -> None:
        """Compare multiple implementations of the same functionality.
        
        Args:
            implementations: Dictionary of name -> function pairs
            args: Arguments to pass to each function
            kwargs: Keyword arguments to pass to each function
            iterations: Number of iterations for each benchmark
        """
        print("\n=== Performance Comparison ===")
        
        results = {}
        for name, func in implementations.items():
            results[name] = self.benchmark_function(
                func, args, kwargs, name, iterations
            )
        
        # Find fastest implementation
        fastest_name = min(results.keys(), key=lambda k: results[k]['mean'])
        fastest_time = results[fastest_name]['mean']
        
        print(f"\nFastest: {fastest_name} ({fastest_time:.4f}s)")
        print("Relative performance:")
        
        for name, stats in results.items():
            ratio = stats['mean'] / fastest_time
            print(f"  {name}: {ratio:.2f}x slower" if ratio > 1 else f"  {name}: baseline")
    
    def profile_sparse_layer(self, layer, input_data: np.ndarray, 
                            k_values: List[int]) -> Dict[int, Dict[str, float]]:
        """Profile sparse layer with different k values.
        
        Args:
            layer: Sparse layer instance
            input_data: Input data
            k_values: List of k values to test
            
        Returns:
            Performance results for each k value
        """
        results = {}
        
        print(f"\n=== Sparse Layer Profiling ===")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {layer.weights.shape[1]}")
        
        for k in k_values:
            layer.num_k_sparse = k
            with timer(f"k={k}") as timing:
                output = layer.get_output(input_data)
                sparsity = np.count_nonzero(output) / output.size
            
            results[k] = {
                "duration": timing["duration"],
                "sparsity": sparsity,
                "non_zeros": np.count_nonzero(output),
                "total_elements": output.size
            }
            
            print(f"  k={k}: {timing['duration']:.4f}s, "
                  f"sparsity: {sparsity:.3f}, "
                  f"non-zeros: {np.count_nonzero(output)}/{output.size}")
        
        return results
    
    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self.results.clear()
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to file.
        
        Args:
            filename: Output filename
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, times in self.results.items():
            serializable_results[name] = {
                "times": times,
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times)
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")