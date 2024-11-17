import numpy as np
from ..utils.initialize_params import init_lg_zeros
from ..logistic_regression.optimize import optimize
from ..logistic_regression.predict import predict

class RegressionModel:
    def __init__(self, num_iterations=1000, learning_rate=0.01, verbose=False):
        """
        Initialize the logistic regression model.

        Parameters
        ----------
        learning_rate : float
            Learning rate for gradient descent.
        num_iterations : int
            Number of iterations for optimization.
        verbose : bool
            Whether to print progress during training.
        """
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.w = None
        self.b = None
        self.costs = []
    
    def fit(self, X, Y):
        """
        Train the logistic regression model.

        Parameters
        ----------
        X : np.ndarray
            Training input data of shape (num_features, num_samples).
        Y : np.ndarray
            Training labels of shape (1, num_samples).
        """
        self.w, self.b = init_lg_zeros(X.shape[0])
        params, _, cost_iter = optimize(
            self.w, self.b, X, Y,
            self.num_iterations, self.learning_rate,
            self.verbose
        )
        self.w, self.b = params["w"], params["b"]
        self.costs = cost_iter
    
    def predict(self, X):
        """
        Predict binary labels for input data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (num_features, num_samples).

        Returns
        -------
        Y_pred : np.ndarray
            Predicted labels of shape (1, num_samples).
        """
        return predict(self.w, self.b, X)
    
    def evaluate(self, X, Y):
        """
        Evaluate model accuracy on a given dataset.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (num_features, num_samples).
        Y : np.ndarray
            Ground truth labels of shape (1, num_samples).

        Returns
        -------
        acc : float
            Accuracy of predictions as a percentage.
        """
        Y_pred = self.predict(X)
        acc = 100 - np.mean(np.abs(Y_pred - Y)) * 100
        return acc
    
    def get_model_info(self):
        """
        Return a dictionary containing some information about the model.

        Returns
        -------
        model_info : dict
            Contains model parameters and training details.
        """
        model_info = {
            "costs": self.costs,
            "w": self.w,
            "b": self.b,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations
        }
        return model_info