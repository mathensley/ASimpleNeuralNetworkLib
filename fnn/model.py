import numpy as np
from ..utils.initialize_params import init_params, init_params_n_layers
from ..utils.cost import compute_cost, compute_cost_gradient
from .propagate import forward_propagation, backward_propagation, forward_propagation_n_layers, backward_propagation_n_layers
from .optimize import optimize
from .predict import predict_two_layer_model, predict_n_layer_model

def two_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations=2000, learning_rate=0.005, verbose=False):
    """
    Implements a two-layer neural network.
    
    Parameters
    ----------
    X_train : training set features, shape (n_x, number of training examples)
    Y_train : training set labels, shape (1, number of training examples)
    X_test : test set features, shape (n_x, number of test examples)
    Y_test : test set labels, shape (1, number of test examples)
    layers_dims : tuple containing the dimensions of the layers (n_x, n_h, n_y)
    num_iterations : number of iterations for optimization loop
    learning_rate : learning rate for gradient descent update rule
    verbose : if True, prints cost every 100 iterations and the final training and test accuracy
    
    Returns
    -------
    model_info : dictionary containing parameters, costs, train/test predictions, learning rate, and iterations
    """
    np.random.seed(1) # for testing
    cost_iter = []
    (n_x, n_h, n_y) = layers_dims

    params = init_params(n_x, n_h, n_y)
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    for i in range(num_iterations):
        A1, cache1 = forward_propagation(X_train, W1, b1, "relu")
        A2, cache2 = forward_propagation(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y_train)

        dA2 = compute_cost_gradient(A2, Y_train) # initializing backpropagation (post activation gradient of the L layer)
        dA1, dW2, db2 = backward_propagation(dA2, cache2, "sigmoid")
        _, dW1, db1 = backward_propagation(dA1, cache1, "relu")

        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        params = optimize(params, gradients, learning_rate)
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]

        if verbose and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after the " + str(i) + "th iteration: " + str(cost))
        if i % 100 == 0 or i == num_iterations:
            cost_iter.append(cost)
    
    Y_pred_train = predict_two_layer_model(params, X_train)
    Y_pred_test = predict_two_layer_model(params, X_test)

    model_info = {
        "costs": cost_iter,
        "params": params,
        "Y_pred_train": Y_pred_train,
        "Y_pred_test": Y_pred_test,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    if verbose:
        train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
        test_acc = 100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100
        print(f"Training accuracy (approx.): {train_acc:.2f}%")
        print(f"Test accuracy (approx.): {test_acc:.2f}%")
    
    return model_info

def n_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations=2000, learning_rate=0.0075, verbose=False):
    """
    Implements an n-layer neural network

    Parameters
    ----------
    X_train : training set features, shape (n_x, number of training examples)
    Y_train : training set labels, shape (1, number of training examples)
    X_test : test set features, shape (n_x, number of test examples)
    Y_test : test set labels, shape (1, number of test examples)
    layers_dims : list containing the dimensions of each layer in the network
    num_iterations : number of iterations for optimization loop
    learning_rate : learning rate for gradient descent update rule
    verbose : if True, prints cost every 100 iterations and the final training and test accuracy
    
    Returns
    -------
    model_info : dictionary containing parameters, costs, train/test predictions, learning rate, and iterations
    """
    np.random.seed(1) # for testing
    cost_iter = []

    params = init_params_n_layers(layers_dims)

    for i in range(num_iterations):
        AL, caches = forward_propagation_n_layers(X_train, params)

        cost = compute_cost(AL, Y_train)

        gradients = backward_propagation_n_layers(AL, Y_train, caches)

        params = optimize(params, gradients, learning_rate)

        if verbose and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after the " + str(i) + "th iteration: " + str(cost))
        if i % 100 == 0 or i == num_iterations:
            cost_iter.append(cost)
    
    Y_pred_train = predict_n_layer_model(params, X_train)
    Y_pred_test = predict_n_layer_model(params, X_test)

    model_info = {
        "costs": cost_iter,
        "params": params,
        "Y_pred_train": Y_pred_train,
        "Y_pred_test": Y_pred_test,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    if verbose:
        train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
        test_acc = 100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100
        print(f"Training accuracy (approx.): {train_acc:.2f}%")
        print(f"Test accuracy (approx.): {test_acc:.2f}%")
    
    return model_info