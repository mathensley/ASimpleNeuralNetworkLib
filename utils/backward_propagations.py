import numpy as np
from .activation_functions import sigmoid, tanh, relu

def sigmoid_backward(dA, activation_cache):
    """
    Computes the backward propagation for sigmoid activation.

    Parameters
    ----------
    dA : post-activation gradient
    activation_cache : the Z value computed during forward pass

    Returns
    -------
    dZ : gradient of the cost with respect to Z, of the same shape as dA
    """
    Z = activation_cache
    s, _ = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ

def tanh_backward(dA, activation_cache):
    """
    Computes the backward propagation for tanh activation.

    Parameters
    ----------
    dA : post-activation gradient
    activation_cache : the Z value computed during forward pass

    Returns
    -------
    dZ : gradient of the cost with respect to Z, of the same shape as dA
    """
    Z = activation_cache
    t, _ = tanh(Z)
    dZ = dA * (1 - (t**2))
    return dZ

def relu_backward(dA, activation_cache):
    """
    Computes the backward propagation for ReLU activation.

    Parameters
    ----------
    dA : post-activation gradient
    activation_cache : the Z value computed during forward pass

    Returns
    -------
    dZ : gradient of the cost with respect to Z, of the same shape as dA
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    #dZ = dA * (Z > 0)
    return dZ