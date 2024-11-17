import numpy as np

def init_lg_zeros(dim):
    """
    Initialize the weights and bias with zeros for a logistic regression model.
    
    Parameters
    ----------
    dim : int
        The number of features (input dimensions)
    
    Returns
    -------
    w : numpy array of shape (dim, 1)
        Weight vector initialized to zeros for each feature
    b : float
        Bias term initialized to zero
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

def init_params(n_x, n_h, n_y):
    """
    Initialize parameters for a two-layer neural network.

    Parameters
    ----------
    n_x : number of input units (features)
    n_h : number of units in the hidden layer
    n_y : number of output units (for binary classification, typically 1)

    Returns
    -------
    params : dictionary containing initialized parameters:
        W1 (weight matrix for hidden layer, of shape (n_h, n_x), initialized with small random values);
        b1 (bias vector for hidden layer, of shape (n_h, 1), initialized to zeros);
        W2 (weight matrix for output layer, of shape (n_y, n_h), initialized with small random values);
        b2 (bias vector for output layer, of shape (n_y, 1), initialized to zeros)
    """
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    return params

def init_params_n_layers(layer_dims):
    """
    Initialize parameters for an n-layer neural network.

    Parameters
    ----------
    layer_dims : list containing the dimensions of each layer in the network, where
                  layer_dims[0] is the number of inputs, layer_dims[1] is the number of hidden units of the first hidden layer, and so on.

    Returns
    -------
    params : dictionary containing the initialized weights and biases for each layer:
              W1, b1, ..., WL-1, bL-1 where L is the total number of layers.
    """
    np.random.seed(1)

    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1]) # Xavier initialization
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))
    
    return params