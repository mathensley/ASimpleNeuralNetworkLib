import numpy as np
from ..utils.activation_functions import sigmoid

def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned parameters (w, b) of logistic regression model
    """
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A, _ = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_pred[0, i] = 1
        else:
            Y_pred[0, i] = 0
    
    return Y_pred