import numpy as np
from ..utils.initialize_params import init_params
from ..utils.cost import compute_cost, compute_cost_gradient
from ..fnn.propagate import forward_propagation, backward_propagation
from ..fnn.optimize import optimize
from ..fnn.predict import predict_two_layer_model

class TwoLayerNNModel:
    def __init__(self, layers_dims, num_iterations=2000, learning_rate=0.005, verbose=False):
        """
        Initialize the Two-Layer neural network model.

        Parameters
        ----------
        layers_dims : list
            Dimensions of the layers (n_x, n_h, n_y).
        num_iterations : int
            Number of iterations for optimization.
        learning_rate : float
            Learning rate for gradient descent.
        verbose : bool
            Whether to print progress during training.
        """
        self.layers_dims = layers_dims
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.costs = []
        self.params = None
    
    def fit(self, X, Y):
        """
        Train the two-layer neural network model.

        Parameters
        ----------
        X : np.ndarray
            Training input data of shape (n_x, num_samples).
        Y : np.ndarray
            Training labels of shape (n_y, num_samples).
        """
        np.random.seed(1)  # for reproducibility

        (n_x, n_h, n_y) = self.layers_dims
        self.params = init_params(n_x, n_h, n_y)
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        for i in range(self.num_iterations):
            A1, cache1 = forward_propagation(X, W1, b1, "relu")
            A2, cache2 = forward_propagation(A1, W2, b2, "sigmoid")

            cost = compute_cost(A2, Y)

            dA2 = compute_cost_gradient(A2, Y) # initializing backpropagation (post activation gradient of the L layer)
            dA1, dW2, db2 = backward_propagation(dA2, cache2, "sigmoid")
            _, dW1, db1 = backward_propagation(dA1, cache1, "relu")

            gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

            self.params = optimize(self.params, gradients, self.learning_rate)
            W1, b1 = self.params["W1"], self.params["b1"]
            W2, b2 = self.params["W2"], self.params["b2"]

            if self.verbose and i % 100 == 0 or i == self.num_iterations - 1:
                print("Cost after the " + str(i) + "th iteration: " + str(cost))
            if i % 100 == 0 or i == self.num_iterations:
                self.costs.append(cost)
    
    def predict(self, X):
        """
        Predict binary labels using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_x, num_samples).

        Returns
        -------
        Y_pred : np.ndarray
            Predicted labels of shape (n_y, num_samples).
        """
        return predict_two_layer_model(self.params, X)
    
    def evaluate(self, X, Y):
        """
        Evaluate model accuracy on a given dataset.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_x, num_samples).
        Y : np.ndarray
            Ground truth labels of shape (n_y, num_samples).

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
            "params": self.params,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations
        }
        return model_info