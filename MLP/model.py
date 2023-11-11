from activations import *
from utils import *
from losses import *


def init_params(layers_dims, initialization="default"):
    """
    Initializes the parameters (weights and biases) for a neural network.

    Parameters:
    layers_dims (list): List of integers representing the dimensions of each layer in the neural network.
    initialization (str): Initialization method to use for parameter initialization.
        - "default": Random initialization using a uniform distribution between -0.5 and 0.5.
        - "xavier": Xavier/Glorot initialization for better training convergence.
        - "he": He initialization for training deeper networks.

    Returns:
    params (dict): A dictionary containing the initialized parameters for each layer.
    """
    params = {}

    for layer in range(1, len(layers_dims)):
        input_dim = layers_dims[layer - 1]
        output_dim = layers_dims[layer]

        # Xavier weight initialization
        if initialization == "xavier":
            params["W" + str(layer)] = np.random.uniform(
                -np.sqrt(6 / (input_dim + output_dim)),
                np.sqrt(6 / (input_dim + output_dim)),
                (output_dim, input_dim),
            )
            params["b" + str(layer)] = np.random.uniform(
                -np.sqrt(6 / (input_dim + output_dim)),
                np.sqrt(6 / (input_dim + output_dim)),
                (output_dim, 1),
            )
        # He weight initialization
        elif initialization == "he":
            params["W" + str(layer)] = np.random.normal(
                0, np.sqrt(2 / input_dim), (output_dim, input_dim)
            )
            params["b" + str(layer)] = np.random.normal(
                0, np.sqrt(2 / input_dim), (output_dim, 1)
            )

        # Default initialization uniform distribution
        else:
            params["W" + str(layer)] = np.random.uniform(
                -0.5, 0.5, (output_dim, input_dim)
            )
            params["b" + str(layer)] = np.random.uniform(-0.5, 0.5, (output_dim, 1))
    return params

def forward_prop(X, params,activation="relu",dropout_prob=0):
    """
    Perform forward propagation through the neural network to compute activations.

    Parameters:
        X (numpy.ndarray): Input data of shape (input_size, m), where input_size is the number of features and m is the number of examples.
        params (dict): Dictionary containing network parameters including weights (W) and biases (b) for each layer.
        dropout_prob (float): Dropout probability for applying dropout regularization to hidden layers (default 0).

    Returns:
        activations (dict): Dictionary containing computed activations for each layer.
        dropout_masks (dict): Dictionary containing dropout masks applied to hidden layers.
    """
    # Get the number of layers directly from the length of W parameters
    L = len(params) // 2
    activations = {}
    activations["A0"] = X # Input activations
    dropout_masks = {} # Dictionary to store dropout masks

    for l in range(1, L):
        # Calculate Z and A for intermediate layers using ReLU activation
        activations["Z" + str(l)] = np.dot(params["W" + str(l)], activations["A" + str(l - 1)]) + params["b" + str(l)]

        if activation == 'tanh':
            activations["A" + str(l)] = tanh(activations["Z" + str(l)])  # Tanh activation
        elif activation == 'sigmoid':
            activations["A" + str(l)] = 1 / (1 + np.exp(-activations["Z" + str(l)]))  # Sigmoid activation
        else :
            activations["A" + str(l)] = relu(activations["Z" + str(l)])  # ReLU activation

        if l < L:
            dropout_mask = np.random.rand(*activations["A" + str(l)].shape) < (1 - dropout_prob)  # Inverted dropout mask
            activations["A" + str(l)] *= dropout_mask
            activations["A" + str(l)] /= (1 - dropout_prob)  # Scale to maintain expected value
            dropout_masks["D" + str(l)] = dropout_mask

    # Calculate Z and A for the output layer using softmax activation
    activations["Z" + str(L)] = np.dot(params["W" + str(L)], activations["A" + str(L - 1)]) + params["b" + str(L)]
    exp_scores = np.exp(activations["Z" + str(L)])
    activations["A" + str(L)] = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)  # Softmax activation

    return activations,dropout_masks


def back_prop(activations, params, Y,dropout_masks,activation="relu"):
    """
    Perform backpropagation to compute gradients of the loss with respect to parameters.

    Parameters:
        activations (dict): Dictionary containing computed activations for each layer during forward propagation.
        params (dict): Dictionary containing network parameters including weights (W) and biases (b) for each layer.
        Y (numpy.ndarray): True labels (ground truth) of shape (1, m), where m is the number of examples.
        dropout_masks (dict): Dictionary containing dropout masks applied to hidden layers.

    Returns:
        grads (dict): Dictionary containing computed gradients of the loss with respect to parameters.
    """
    L = len(params) // 2
    one_hot_Y = one_hot_encode(Y)
    m = one_hot_Y.shape[1]

    derivatives = {}
    grads = {}

    # for layer L
    derivatives["dZ" + str(L)] = activations["A" + str(L)] - one_hot_Y
    grads["dW" + str(L)] = (
            1 / m * np.dot(derivatives["dZ" + str(L)], activations["A" + str(L - 1)].T)
    )
    grads["db" + str(L)] = 1 / m * np.sum(derivatives["dZ" + str(L)])

    # for layers L-1 to 1
    for l in reversed(range(1, L)):
        if activation == 'relu':
            _activation_derivative = relu_derivative
        elif activation == 'tanh':
            _activation_derivative = tanh_derivative
        elif activation == 'sigmoid':
            _activation_derivative = sigmoid_derivative

        derivatives["dZ" + str(l)] = np.dot(
            params["W" + str(l + 1)].T, derivatives["dZ" + str(l + 1)]
        ) * _activation_derivative(activations["Z" + str(l)])

        # apply dropout mask
        if l < L:
            derivatives["dZ" + str(l)] *= dropout_masks["D" + str(l)]

        grads["dW" + str(l)] = (
                1 / m * np.dot(derivatives["dZ" + str(l)], activations["A" + str(l - 1)].T)
        )
        grads["db" + str(l)] = (
                1 / m * np.sum(derivatives["dZ" + str(l)], axis=1, keepdims=True)
        )

    return grads


def update_params(params, grads, alpha):
    """
    Update network parameters using gradient descent optimization.

    Parameters:
        params (dict): Dictionary containing network parameters including weights (W) and biases (b) for each layer.
        grads (dict): Dictionary containing gradients of the loss with respect to parameters.
        alpha (float): Learning rate, controlling the step size of parameter updates.

    Returns:
        params_updated (dict): Dictionary containing updated network parameters.
    """
    # number of layers
    L = len(params) // 2

    params_updated = {}
    for l in range(1, L + 1):
        params_updated["W" + str(l)] = (
                params["W" + str(l)] - alpha * grads["dW" + str(l)]
        )
        params_updated["b" + str(l)] = (
                params["b" + str(l)] - alpha * grads["db" + str(l)]
        )

    return params_updated

def train(X, Y, params, max_iter=10, learning_rate=0.1,dropout_prob=0,activation="relu"):
    """
    Trains a neural network model using gradient descent optimization.

    Parameters:
    X (numpy.ndarray): Input data of shape (input_size, num_samples).
    Y (numpy.ndarray): Ground truth labels of shape (1, num_samples).
    params (dict): Dictionary containing the parameters of the neural network.
                   Keys are "W1", "b1", ..., "WL", "bL" where L is the number of layers.
    max_iter (int, optional): Maximum number of training iterations. Default is 10.
    learning_rate (float, optional): Learning rate for gradient descent. Default is 0.1.
    dropout_prob (float, optional): Dropout probability for regularization. Default is 0.
    activation (str, optional): Activation function to use in hidden layers. Default is "relu".

    Returns:
    tuple: A tuple containing updated parameters, list of accuracies per iteration, and list of losses per iteration.
    """
    # Initialize parameters Wl, bl for layers l=1,...,L
    L = len(params) // 2
    accuracies = []
    losses = []

    for iteration in range(1, max_iter + 1):
        # Forward propagation
        activations,dropout_mask = forward_prop(X, params,activation,dropout_prob)

        # Make predictions
        Y_hat = get_predictions(activations["A" + str(L)])

        # Compute accuracy
        accuracy = get_accuracy(Y_hat, Y)
        accuracies.append(accuracy)

        # Compute loss (cross-entropy)
        loss = cross_entropy(one_hot_encode(Y), activations["A" + str(L)])
        losses.append(loss)

        # Backpropagation
        gradients = back_prop(activations, params, Y,dropout_mask,activation)

        # Update parameters
        params = update_params(params, gradients, learning_rate)

        # Print progress
        if iteration ==1 or (iteration%5) == 0:
            print("Iteration {}: Accuracy = {}, Loss = {}".format(iteration, accuracy, loss))

    return params,accuracies,losses

def test(X_test, Y_test, params,activation):
    """
    Evaluates the trained neural network model on test data.

    Parameters:
    X_test (numpy.ndarray): Test input data of shape (input_size, num_samples).
    Y_test (numpy.ndarray): Ground truth labels for test data of shape (1, num_samples).
    params (dict): Dictionary containing the parameters of the neural network.
                   Keys are "W1", "b1", ..., "WL", "bL" where L is the number of layers.
    activation (str): Activation function used in the hidden layers during forward propagation.

    Returns:
    tuple: A tuple containing test accuracy and loss.
    """
    L = len(params) // 2

    # Forward propagation
    activations,_ = forward_prop(X_test, params,activation,0)

    # Make predictions
    Y_hat = get_predictions(activations["A" + str(L)])

    # Compute test accuracy
    test_accuracy = get_accuracy(Y_hat, Y_test)
    loss = cross_entropy(one_hot_encode(Y_test), activations["A" + str(L)])

    return test_accuracy,loss