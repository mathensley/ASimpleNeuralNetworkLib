import copy

def optimize(params, gradients, learning_rate):
    """
    Optimize parameters using gradient descent.
    """
    params = copy.deepcopy(params)
    L = len(params) // 2

    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate * gradients["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate * gradients["db" + str(l+1)]
    
    return params