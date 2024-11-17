from .propagate import forward_propagation, forward_propagation_n_layers

def predict_two_layer_model(params, X):
    """
    Make predictions for each example in X using learned parameters of a two-layer model.
    
    Parameters
    ----------
    params : dictionary containing parameters W1, b1, W2, b2
    X : data set for which predictions need to be made, shape (n_x, number of examples)
    
    Returns
    -------
    Y_pred : predictions for each example in X (1 or 0)
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    A1, _ = forward_propagation(X, W1, b1, "relu")
    A2, _ = forward_propagation(A1, W2, b2, "sigmoid")

    Y_pred = (A2 > 0.5).astype(int)
    return Y_pred

def predict_n_layer_model(params, X):
    """
    Make predictions for each example in X using learned parameters in an n-layer model.
    
    Parameters
    ----------
    params : dictionary containing the parameters of the model (weights and biases for each layer)
    X : dataset for which predictions need to be made, shape (n_x, number of examples)
    
    Returns
    -------
    Y_pred : predictions for each example in X (1 or 0)
    """
    AL, _ = forward_propagation_n_layers(X, params)

    Y_pred = (AL > 0.5).astype(int)
    return Y_pred