import numpy as np
from ..utils.activation_functions import sigmoid
from ..utils.cost import compute_cost

def forward_propagation(w, b, X, Y):
    """
    Perform forward propagation and calculate cost
    """
    A, _ = sigmoid(np.dot(w.T, X) + b)
    cost = compute_cost(A, Y)
    return A, cost

def backward_propagation(A, X, Y):
    """
    Perform backward propagation
    """
    m = Y.shape[1]
    dw = np.dot(X, (A-Y).T) / m
    db = np.sum(A - Y) / m

    gradients = {"dw": dw,
                 "db": db}
    return gradients