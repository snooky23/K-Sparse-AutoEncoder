import numpy as np


# activation function
def sigmoid_function(signal, derivative=False):
    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1.0 - signal)
    else:
        # Return the activation signal
        return 1.0 / (1.0 + np.exp(-signal))


def ReLU_function(signal, derivative=False):
    if derivative:
        return (signal > 0).astype(float)
    else:
        # Return the activation signal
        return np.maximum(0, signal)


def tanh_function(signal, derivative=False):
    # Calculate activation signal
    if derivative:
        # Return the partial derivation of the activation function
        return 1 - np.power(signal, 2)
    else:
        # Return the activation signal
        return np.tanh(signal)


def softmax_function(signal):
    # Calculate activation signal
    e_x = np.exp(signal - np.max(signal, axis=1, keepdims=True))
    signal = e_x / np.sum(e_x, axis=1, keepdims=True)
    return signal

# end activation function