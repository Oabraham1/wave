# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

WAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MNISTNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def initialize_like_wave(model, seed=42):
    rng = np.random.RandomState(seed)

    input_dim, hidden_dim = 784, 128
    output_dim = 10

    scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
    W1 = (rng.randn(input_dim, hidden_dim) * scale1).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)

    scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
    W2 = (rng.randn(hidden_dim, output_dim) * scale2).astype(np.float32)
    b2 = np.zeros(output_dim, dtype=np.float32)

    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(W1.T))
        model.fc1.bias.copy_(torch.from_numpy(b1))
        model.fc2.weight.copy_(torch.from_numpy(W2.T))
        model.fc2.bias.copy_(torch.from_numpy(b2))


def generate_batch_indices(n_samples, epochs, seed=7):
    rng = np.random.RandomState(seed)
    return [rng.permutation(n_samples) for _ in range(epochs)]


def train(
    epochs=5,
    batch_size=32,
    lr=0.1,
    hidden_dim=128,
    eval_every=100,
    max_batches=None,
    device_name="cpu",
    shuffle_seed=7,
):
    data_dir = os.path.join(WAVE_ROOT, "data")
    print("Loading MNIST...")

    train_ds = datasets.MNIST(
        data_dir, train=True, download=False,
        transform=transforms.ToTensor()
    )
    test_ds = datasets.MNIST(
        data_dir, train=False, download=False,
        transform=transforms.ToTensor()
    )

    train_images = np.array([img.numpy().flatten() for img, _ in train_ds], dtype=np.float32)
    train_labels = np.array([label for _, label in train_ds], dtype=np.int64)
    test_images = np.array([img.numpy().flatten() for img, _ in test_ds], dtype=np.float32)
    test_labels = np.array([label for _, label in test_ds], dtype=np.int64)

    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    device = torch.device(device_name)
    print(f"\nInitializing network: 784 -> {hidden_dim} -> 10")
    print(f"  Backend: Native PyTorch ({device_name})")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Shuffle seed: {shuffle_seed}")

    model = MNISTNet(784, hidden_dim, 10).to(device)
    initialize_like_wave(model, seed=42)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epoch_indices = generate_batch_indices(len(train_images), epochs, seed=shuffle_seed)

    print("\nTraining started (Native PyTorch)")
    results = []
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        indices = epoch_indices[epoch]

        for batch_start in range(0, len(train_images), batch_size):
            if max_batches is not None and n_batches >= max_batches:
                break

            batch_indices = indices[batch_start:batch_start + batch_size]
            if len(batch_indices) < batch_size:
                continue

            X_batch = torch.from_numpy(train_images[batch_indices]).to(device)
            y_batch = torch.from_numpy(train_labels[batch_indices]).to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % eval_every == 0:
                avg_loss = epoch_loss / n_batches
                elapsed = time.time() - epoch_start
                print(
                    f"  Epoch {epoch + 1} | Batch {n_batches} | "
                    f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        model.eval()
        with torch.no_grad():
            test_tensor = torch.from_numpy(test_images).to(device)
            test_preds = model(test_tensor).argmax(dim=1).cpu().numpy()
        test_acc = np.mean(test_preds == test_labels)

        with torch.no_grad():
            train_tensor = torch.from_numpy(train_images[:1000]).to(device)
            train_preds = model(train_tensor).argmax(dim=1).cpu().numpy()
        train_acc = np.mean(train_preds == train_labels[:1000])

        result = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "epoch_time": epoch_time,
            "batches": n_batches,
        }
        results.append(result)

        print(
            f"\nEpoch {epoch + 1}/{epochs} complete | "
            f"Loss: {avg_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Test acc: {test_acc:.4f} | "
            f"Time: {epoch_time:.1f}s\n"
        )

    total_time = time.time() - total_start
    print(f"\nTraining complete (Native PyTorch {device_name})")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final test accuracy: {results[-1]['test_acc']:.4f}")
    print(f"Final train accuracy: {results[-1]['train_acc']:.4f}")

    return model, results


if __name__ == "__main__":
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_batches = int(sys.argv[3]) if len(sys.argv) > 3 else None
    device_name = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    shuffle_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7

    model, results = train(
        epochs=epochs,
        batch_size=batch_size,
        lr=0.1,
        hidden_dim=128,
        eval_every=50,
        max_batches=max_batches,
        device_name=device_name,
        shuffle_seed=shuffle_seed,
    )
