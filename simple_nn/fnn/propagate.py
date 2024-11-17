import numpy as np
from ..utils.activation_functions import sigmoid, tanh, relu
from ..utils.backward_propagations import sigmoid_backward, tanh_backward, relu_backward
from ..utils.cost import compute_cost_gradient

def forward_propagation_linear(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def forward_propagation(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = forward_propagation_linear(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "tanh":
        Z, linear_cache = forward_propagation_linear(A_prev, W, b)
        A, activation_cache = tanh(Z)
    elif activation == "relu":
        Z, linear_cache = forward_propagation_linear(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_propagation_n_layers(X, params):
    caches = []
    A = X
    L = len(params) // 2 # computes the number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        A, cache = forward_propagation(A_prev, params["W" + str(l)], params["b" + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = forward_propagation(A, params["W" + str(L)], params["b" + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches

def backward_propagation_linear(dZ, cache):
    A_prev, W, _ = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backward_propagation(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = backward_propagation_linear(dZ, linear_cache)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = backward_propagation_linear(dZ, linear_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = backward_propagation_linear(dZ, linear_cache)
    return dA_prev, dW, db

def backward_propagation_n_layers(AL, Y, caches):
    gradients = {}
    L = len(caches) # the number of layers of the neural network
    Y = Y.reshape(AL.shape)

    dAL = compute_cost_gradient(AL, Y) # initializing backpropagation (post activation gradient of the L layer)

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = backward_propagation(dAL, current_cache, "sigmoid")
    gradients["dA" + str(L-1)] = dA_prev_temp
    gradients["dW" + str(L)] = dW_temp
    gradients["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_propagation(dA_prev_temp, current_cache, "relu")
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l+1)] = dW_temp
        gradients["db" + str(l+1)] = db_temp
    
    return gradients