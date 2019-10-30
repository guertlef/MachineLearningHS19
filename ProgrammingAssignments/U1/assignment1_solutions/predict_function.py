from tanh import tanh
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.

    """

    preds = None
    accuracy = None
    #######################################################################
    # TODO:                                                               #
    # Compute predictions on X using the parameters theta.                #
    # If y is provided compute the accuracy of the classifier as well.    #
    #                                                                     #
    #######################################################################
    
    #preds = np.around(tanh(np.dot(X, theta)))
    h = tanh(np.dot(X, theta))
    preds = np.where(h > 0, 1, -1)
    
    if (y.any != None):
        accuracy = (np.sum(preds==y)/y.shape[0]) * 100
    

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return preds, accuracy