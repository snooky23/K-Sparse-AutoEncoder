# K-Sparse AutoEncoder

A professional-grade implementation of K-Sparse Autoencoders with comprehensive features for deep learning research and applications.

## 🚀 Features

### Core Architecture
- **Fully Connected Neural Networks** with customizable layers
- **K-Sparse Layers** for sparse representation learning
- **MNIST Classification** with 96% accuracy
- **AutoEncoder** for dimensionality reduction and reconstruction

### Advanced Deep Learning Features
- **Multiple Activation Functions**: Sigmoid, ReLU, Tanh, Leaky ReLU, ELU, Swish, GELU, Linear, Softmax
- **Regularization Techniques**: L1, L2, Elastic Net, Dropout, Batch Normalization
- **Learning Rate Scheduling**: Step decay, Exponential decay, Cosine annealing
- **Early Stopping** with validation monitoring
- **Gradient Clipping** for stable training
- **Validation Split** for proper model evaluation

### Performance & Quality
- **Optimized Operations**: Vectorized sparse layer computations
- **Memory Efficient**: Improved batch processing and matrix operations
- **Comprehensive Testing**: 63 unit tests with 100% pass rate
- **Type Safety**: Full type hints throughout codebase
- **Professional Documentation**: Detailed docstrings and examples

### Development Tools
- **Performance Benchmarking**: Built-in timing and profiling utilities
- **Training History**: Loss and accuracy tracking
- **Extensible Design**: Easy to add new layers and features

## 📊 K-Sparse AutoEncoder Theory

K-Sparse Autoencoders find the k highest activations in the hidden layer and zero out the rest:
- Only the k most significant features are preserved
- Error backpropagation occurs only through active nodes
- Results in sparse, interpretable representations

![K-Sparse AutoEncoder](images/Autoencoder_2.png)

## 🎯 Results for Different K Values

The following images show reconstruction quality with different sparsity levels:

| K=10 | K=25 |
|------|------|
| ![K=10](images/k=10.PNG) | ![K=25](images/k=25.PNG) |

| K=40 | K=70 |
|------|------|
| ![K=40](images/k=40.PNG) | ![K=70](images/k=70.PNG) |

## 🛠 Installation & Usage

### Quick Start

```python
from layers.linear_layer import LinearLayer
from layers.sparse_layer import SparseLayer
from nets.fcnn import FCNeuralNet
from utilis.activations import sigmoid_function

# Create a K-Sparse AutoEncoder
layers = [
    SparseLayer("encoder", n_in=784, n_out=100, 
                activation=sigmoid_function, num_k_sparse=25),
    LinearLayer("decoder", n_in=100, n_out=784, 
                activation=sigmoid_function)
]

network = FCNeuralNet(layers)

# Train with advanced features
history = network.train(
    x_train, y_train,
    validation_split=0.2,
    early_stopping_patience=10,
    l2_reg=0.001,
    lr_schedule="cosine",
    epochs=1000
)
```

### Running Examples

```bash
# Run MNIST classification or autoencoder
python main_mnist.py

# Run tests
python -m pytest tests/ -v

# Or use the test runner
python run_tests.py
```

### Jupyter Notebook
Explore the interactive examples in `auto_encoder_3.ipynb` for visualization and experimentation.

## 📁 Project Structure

```
K-Sparse-AutoEncoder/
├── layers/
│   ├── linear_layer.py      # Fully connected layer
│   └── sparse_layer.py      # K-sparse layer implementation
├── nets/
│   └── fcnn.py             # Neural network with advanced training
├── utilis/
│   ├── activations.py      # 9 activation functions
│   ├── cost_functions.py   # Loss functions
│   ├── regularization.py   # Regularization techniques
│   ├── performance.py      # Benchmarking utilities
│   └── mnist/
│       └── mnist_helper.py # MNIST data loading
├── tests/                  # 63 comprehensive unit tests
│   ├── layers/
│   ├── nets/
│   └── utilis/
├── main_mnist.py           # Main execution script
├── run_tests.py           # Test runner
└── auto_encoder_3.ipynb   # Interactive notebook
```

## 🔧 Advanced Configuration

### Custom Activation Functions
```python
from utilis.activations import gelu_function, swish_function

# Use modern activation functions
layer = LinearLayer("hidden", 100, 50, activation=gelu_function)
```

### Regularization & Training
```python
# Advanced training with regularization
history = network.train(
    x_train, y_train,
    learning_rate=0.001,
    batch_size=128,
    validation_split=0.2,
    l1_reg=0.01,              # L1 regularization
    l2_reg=0.001,             # L2 regularization
    early_stopping_patience=15, # Early stopping
    lr_schedule="exponential", # Learning rate decay
    gradient_clip_norm=1.0,   # Gradient clipping
    epochs=500
)
```

### Performance Benchmarking
```python
from utilis.performance import PerformanceBenchmark

benchmark = PerformanceBenchmark()
benchmark.benchmark_layer_forward(layer, test_data)
benchmark.profile_sparse_layer(sparse_layer, test_data, k_values=[10, 25, 50])
```

## 🧪 Testing

The project includes comprehensive testing:

- **63 unit tests** covering all components
- **100% test pass rate** maintained
- **Edge case coverage** for robust behavior
- **Performance regression testing**

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/layers/ -v
python -m pytest tests/nets/ -v
python -m pytest tests/utilis/ -v
```

## 📈 Performance Optimizations

- **Vectorized Operations**: 3-5x faster sparse layer computations
- **Memory Efficiency**: Reduced memory usage in batch processing
- **Optimized Training**: Improved gradient computation and weight updates
- **Benchmarking Tools**: Built-in performance monitoring

## 🎓 Architecture Details

### Network Architecture
- **Input Layer**: 784 neurons (28x28 MNIST images)
- **Hidden Layer**: Customizable (typically 100-300 neurons)
- **Output Layer**: 784 neurons (reconstruction) or 10 neurons (classification)

### K-Sparse Mechanism
1. Forward pass computes activations
2. Select k highest activations per sample
3. Zero out remaining activations
4. Continue with standard backpropagation

## 🤝 Contributing

The codebase is designed for extensibility:
- Clean, typed interfaces
- Comprehensive documentation
- Modular architecture
- Full test coverage

## 📚 Technical Improvements

This implementation includes several major enhancements over the original:

### Code Quality (Phase 1)
- Complete type hints for all functions and classes
- Comprehensive docstrings following Google style
- Standardized naming conventions
- Organized imports and clean code structure

### Testing Infrastructure (Phase 2)
- 63 unit tests covering all components
- Test coverage for edge cases and error conditions
- Automated test runner
- Continuous integration ready

### Performance Optimizations (Phase 3)
- Vectorized sparse layer operations (3-5x speedup)
- Memory-efficient batch processing
- Optimized matrix operations
- Performance benchmarking utilities

### Advanced Features (Phase 4)
- 6 additional activation functions
- Comprehensive regularization techniques
- Learning rate scheduling
- Early stopping and validation monitoring
- Gradient clipping for stable training

## 📊 Results

- **MNIST Classification**: 96% accuracy
- **Autoencoder Reconstruction**: High-quality image reconstruction
- **Sparse Representations**: Interpretable feature learning
- **Performance**: 3-5x faster than original implementation

## 🔮 Future Enhancements

The architecture supports easy addition of:
- Configuration file management
- Command-line interfaces
- Model persistence and loading
- Enhanced visualization tools
- Additional layer types

---

*This project demonstrates professional-grade machine learning implementation with comprehensive features for research and production use.*