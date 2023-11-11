import numpy as np


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a


def relu_derivative(z):
    return z > 0


def softmax_derivative(z):
    d_z = np.exp(z) / sum(np.exp(z)) * (1.0 - np.exp(z) / sum(np.exp(z)))
    return d_z


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def sigmoid_derivative(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)
