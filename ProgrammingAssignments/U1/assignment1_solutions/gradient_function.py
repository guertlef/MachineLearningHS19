from tanh import tanh
import numpy as np


def gradient_function(theta, X, y):
    """
    Compute gradient for regression w.r.t. to the parameters theta.

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        grad: The gradient of the cost w.r.t. theta

    """

    grad = None
    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of theta.              #
    # Compute the partial derivatives and set grad to the partial         #
    # derivatives of the cost w.r.t. each parameter in theta              #
    #                                                                     #
    #######################################################################
    # if X is vector
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    
    h = tanh(np.dot(X, theta))
    grad = 2*np.dot((h-y)*(1-h*h), X)
    grad /= X.shape[0]
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad