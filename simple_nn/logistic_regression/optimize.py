import copy
from .propagate import forward_propagation, backward_propagation

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, verbose=False):
    """
    Optimize w and b using gradient descent.
    """
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    cost_iter = []

    for i in range(num_iterations):
        A, cost = forward_propagation(w, b, X, Y)
        gradients = backward_propagation(A, X, Y)

        dw, db = gradients["dw"], gradients["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            cost_iter.append(cost)
            if verbose:
                print("Cost after the " + str(i) + "th iteration: " + str(cost))
    
    params = {"w": w,
              "b": b}
    gradients = {"dw": dw,
                 "db": db}
    
    return params, gradients, cost_iter