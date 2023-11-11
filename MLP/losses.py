import numpy as np


def cross_entropy(y_one_hot, y_hat, epsilon=1e-10):
    # clip predictions to avoid values of 0 and 1
    y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)
    # sum on the columns of Y_hat * np.log(Y), then take the mean
    # between the m samples
    entropy = -np.mean(np.sum(y_one_hot * np.log(y_hat), axis=0))
    return entropy
