import numpy as np

def compute_cost(AL, Y):
    """
    Computes the cost function, negative log-likelihood cost (cross-entropy).
    """
    m = Y.shape[1]
    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL))) / m
    return np.squeeze(cost)

def compute_cost_gradient(AL, Y):
    """
    Computes the derivative of the cost function with respect to AL (the output of the final layer).
    
    Parameters
    ----------
    AL : predicted output vector after applying sigmoid function (output of the final layer),
          shape (1, m) where m is the number of examples
    Y : true labels vector (1 for positive class, 0 for negative class), shape (1, m)
    
    Returns
    -------
    dAL : derivative of the cost with respect to AL, shape (1, m)
    """
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    return dAL