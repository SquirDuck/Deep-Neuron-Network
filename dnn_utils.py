import numpy as np

def sigmoid(Z):
    """
    Tính toán hàm sigmoid.
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Tính toán hàm RELU.
    """
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Tính toán backward propagation cho RELU.
    """
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    """
    Tính toán backward propagation cho sigmoid.
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ
