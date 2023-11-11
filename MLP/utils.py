import numpy as np
import matplotlib.pyplot as plt


def normalize_pixels(data):
    return data / 255.0


def shuffle_rows(data):
    # Convert input dataframe to ndarray
    data = np.array(data)
    np.random.shuffle(data)
    return data


def get_predictions(al):
    # get the max index by the columns
    return np.argmax(al, axis=0)


def get_accuracy(y_hat, y):
    return np.sum(y_hat == y) / y.size


def one_hot_encode(y):
    y_one_hot = np.zeros((y.shape[0], y.max() + 1))
    # set to 1 the correct indices
    y_one_hot[np.arange(y.shape[0]), y] = 1
    # transpose
    y_one_hot = y_one_hot.T
    return y_one_hot


def plot_accuracy_and_loss(accuracies, losses, max_iter):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), accuracies, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), losses, marker='o', color='r')
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()
