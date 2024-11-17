import numpy as np

def sigmoid(Z):
    """
    Computes the sigmoid of Z.

    Parameters
    ----------
    Z : numpy array of any shape

    Returns
    -------
    s : sigmoid(Z) of the input value(s), ranging between 0 and 1
    Z : the original input Z (for use in backpropagation)
    """
    s = 1 / (1+np.exp(-Z))
    return s, Z

def tanh(Z):
    """
    Computes the tanh of Z.

    Parameters
    ----------
    Z : numpy array of any shape

    Returns
    -------
    t : tanh(Z), element-wise
    Z : the original input Z (for use in backpropagation)
    """
    t = np.tanh(Z)
    return t, Z

def relu(Z):
    """
    Computes the ReLU of Z.

    Parameters
    ----------
    Z : numpy array of any shape

    Returns
    -------
    r : relu(Z), element-wise where relu(x) = max(0, x)
    Z : the original input Z (for use in backpropagation)
    """
    r = np.maximum(0, Z)
    return r, Z