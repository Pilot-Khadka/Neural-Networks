import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[2], num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def load_mnist_data(batch_size):
    print("Loading MNIST dataset...")

    transform_for_stats = transforms.Compose([transforms.ToTensor()])
    train_set_temp = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform_for_stats
    )

    data_tensor = torch.stack([img_t for img_t, _ in train_set_temp])
    mean = data_tensor.mean()
    std = data_tensor.std()
    print(f"Calculated Mean: {mean.item():.4f}, Std: {std.item():.4f}")

    final_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean.item()], std=[std.item()]),
        ]
    )

    train_set = datasets.MNIST(
        root="./data", train=True, download=False, transform=final_transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=False, transform=final_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            avg_loss_batch = loss.item()
            accuracy_batch = (
                100.0 * predicted.eq(targets).sum().item() / targets.size(0)
            )
            sys.stdout.write(
                f"\r  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {avg_loss_batch:.4f}, "
                f"Batch Accuracy = {accuracy_batch:.2f}%"
            )
            sys.stdout.flush()

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total

    print()
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 20
    batch_size = 128
    input_size = 28 * 28
    hidden_sizes = [512, 256, 128]
    num_classes = 10
    learning_rate = 0.001

    train_loader, test_loader = load_mnist_data(batch_size)
    print(
        f"Training set size: {len(train_loader.dataset)}, Test set size: {len(test_loader.dataset)}"
    )

    model = FullyConnectedNet(input_size, hidden_sizes, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
