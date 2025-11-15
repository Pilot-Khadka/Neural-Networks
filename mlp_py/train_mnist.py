import sys
import gzip
import struct
import numpy as np
import urllib.request
from pathlib import Path

from optim import Adam
from tensor import Tensor
from activations import relu
from nn import Linear, Sequential, Dropout, BatchNorm1d, cross_entropy_loss


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        ratio = min(downloaded / total_size, 1)
        width = 40
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[{bar}] {ratio * 100:5.1f}%")
        sys.stdout.flush()
        if ratio == 1:
            sys.stdout.write("\n")


def read_idx(filename):
    print(f"Reading {filename}...")
    with gzip.open(filename, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))

        if magic == 2051:  # Images
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_items, rows, cols)
        elif magic == 2049:  # Labels
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data  # already 1D array of length num_items
        else:
            raise ValueError(f"Invalid IDX file with magic number {magic}")


def check_and_download_mnist(data_dir="./data/mnist"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    file_info = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    for filename in file_info.values():
        filepath = data_dir / filename
        if not filepath.exists():
            url = base_url + filename
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath, _progress)
        else:
            print(f"{filename} already exists.")

    return data_dir, file_info


def load_mnist(data_dir="./data/mnist"):
    data_dir, file_info = check_and_download_mnist(data_dir)

    X_train = read_idx(data_dir / file_info["train_images"])
    y_train = read_idx(data_dir / file_info["train_labels"])
    X_test = read_idx(data_dir / file_info["test_images"])
    y_test = read_idx(data_dir / file_info["test_labels"])

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def create_batches(X, y, batch_size):
    indices = np.random.permutation(len(X))
    for start_idx in range(0, len(X), batch_size):
        batch_indices = indices[start_idx : start_idx + batch_size]
        yield X[batch_indices], y[batch_indices]


def train_epoch(model, optimizer, X_train, y_train, batch_size):
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, (X_batch, y_batch) in enumerate(
        create_batches(X_train, y_train, batch_size)
    ):
        X_tensor = Tensor(X_batch, requires_grad=True)

        optimizer.zero_grad()

        predictions = model(X_tensor)
        loss = cross_entropy_loss(predictions, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
        pred_classes = np.argmax(predictions.data, axis=1)
        correct += (pred_classes == y_batch).sum()
        total += len(y_batch)
        num_batches += 1

        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Batch {batch_idx + 1}: Loss = {loss.data.item():.4f}, "
                f"Accuracy = {100 * correct / total:.2f}%"
            )

    return total_loss / num_batches, 100 * correct / total


def evaluate(model, X_test, y_test, batch_size):
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    for X_batch, y_batch in create_batches(X_test, y_test, batch_size):
        X_tensor = Tensor(X_batch, requires_grad=False)

        predictions = model(X_tensor)
        loss = cross_entropy_loss(predictions, y_batch)

        total_loss += loss.data.item()
        pred_classes = np.argmax(predictions.data, axis=1)
        correct += (pred_classes == y_batch).sum()
        total += len(y_batch)
        num_batches += 1

    return total_loss / num_batches, 100 * correct / total


def main():
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    input_size = 784
    hidden_sizes = [512, 256, 128]
    num_classes = 10

    model = Sequential(
        Linear(input_size, hidden_sizes[0]),
        BatchNorm1d(hidden_sizes[0]),
        relu,
        Dropout(0.3),
        Linear(hidden_sizes[0], hidden_sizes[1]),
        BatchNorm1d(hidden_sizes[1]),
        relu,
        Dropout(0.3),
        Linear(hidden_sizes[1], hidden_sizes[2]),
        BatchNorm1d(hidden_sizes[2]),
        relu,
        Dropout(0.2),
        Linear(hidden_sizes[2], num_classes),
    )

    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    batch_size = 128

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(
            model, optimizer, X_train, y_train, batch_size
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        test_loss, test_acc = evaluate(model, X_test, y_test, batch_size)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
