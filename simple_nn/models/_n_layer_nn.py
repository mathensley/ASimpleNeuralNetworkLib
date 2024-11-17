import numpy as np
from ..utils.initialize_params import init_params_n_layers
from ..utils.cost import compute_cost
from ..fnn.propagate import forward_propagation_n_layers, backward_propagation_n_layers
from ..fnn.optimize import optimize
from ..fnn.predict import predict_n_layer_model

class NLayerNNModel:
    def __init__(self, layers_dims, num_iterations=2000, learning_rate=0.0075, verbose=False):
        """
        Initialize the N-Layer neural network model.

        Parameters
        ----------
        layers_dims : tuple
            A list specifying the number of units in each layer of the neural network, 
            starting with the input layer and ending with the output layer. E.g., 
            [12288, 8, 4, 2, 1] for a 4-layer network (the input features in this case are the pixels of an image).
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
        Train the n-layer neural network model.

        Parameters
        ----------
        X : np.ndarray
            Training input data of shape (n_x, num_samples).
        Y : np.ndarray
            Training labels of shape (n_y, num_samples).
        """
        np.random.seed(1)  # for reproducibility

        self.params = init_params_n_layers(self.layers_dims)

        for i in range(self.num_iterations):
            AL, caches = forward_propagation_n_layers(X, self.params)

            cost = compute_cost(AL, Y)

            gradients = backward_propagation_n_layers(AL, Y, caches)

            self.params = optimize(self.params, gradients, self.learning_rate)

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
        return predict_n_layer_model(self.params, X)
    
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