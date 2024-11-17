import numpy as np
from ..utils.initialize_params import init_lg_zeros
from .optimize import optimize
from .predict import predict

def train_model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.5, verbose=False):
    """
    Train logistic regression model and provide predictions with performance metrics.

    Returns
    --------
    model_info : dict
        Contains information about the model.
    """

    w, b = init_lg_zeros(X_train.shape[0])
    params, gradients, cost_iter = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, verbose)

    w, b = params["w"], params["b"]

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test = predict(w, b, X_test)

    if verbose:
        train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
        test_acc = 100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100
        print(f"Training accuracy: {train_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
    
    model_info = {"costs": cost_iter,
                  "Y_pred_train": Y_pred_train,
                  "Y_pred_test": Y_pred_test,
                  "w": w,
                  "b": b,
                  "learning_rate": learning_rate,
                  "num_iterations": num_iterations}
    return model_info

