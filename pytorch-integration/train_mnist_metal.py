# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import numpy as np
from torchvision import datasets, transforms

from wave_kernels import ALL_KERNELS
from wave_runtime import compile_kernel
from wave_metal_runner import run_kernel_metal

WAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_mnist(data_dir):
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

    return train_images, train_labels, test_images, test_labels


def initialize_weights(input_dim, hidden_dim, output_dim, seed=42):
    rng = np.random.RandomState(seed)
    scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
    W1 = (rng.randn(input_dim, hidden_dim) * scale1).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)

    scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
    W2 = (rng.randn(hidden_dim, output_dim) * scale2).astype(np.float32)
    b2 = np.zeros(output_dim, dtype=np.float32)

    return W1, b1, W2, b2


class WaveMetalNN:

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr

        self.W1, self.b1, self.W2, self.b2 = initialize_weights(
            input_dim, hidden_dim, output_dim
        )

        self.kernels = {}
        for name, source in ALL_KERNELS.items():
            self.kernels[name] = compile_kernel(name, source)

    def _run(self, kernel_name, buffers, scalars, output_specs, workgroup, grid=(1, 1, 1)):
        return run_kernel_metal(
            self.kernels[kernel_name],
            buffers=buffers,
            scalars=scalars,
            output_specs=output_specs,
            workgroup=workgroup,
            grid=grid,
        )

    def forward(self, X_batch):
        batch_size = X_batch.shape[0]
        M, K, N = batch_size, self.input_dim, self.hidden_dim

        z1 = np.zeros((M, N), dtype=np.float32)
        [z1_flat] = self._run(
            "matmul",
            [X_batch.flatten(), self.W1.flatten(), z1.flatten()],
            [(M, "u32"), (K, "u32"), (N, "u32")],
            [(2, M * N)],
            workgroup=(min(256, M * N), 1, 1),
            grid=(max(1, (M * N + 255) // 256), 1, 1),
        )
        z1 = z1_flat.reshape(M, N)

        z1_biased = np.zeros_like(z1)
        [z1_biased_flat] = self._run(
            "bias_add",
            [z1.flatten(), self.b1, z1_biased.flatten()],
            [(M, "u32"), (N, "u32")],
            [(2, M * N)],
            workgroup=(min(256, M * N), 1, 1),
            grid=(max(1, (M * N + 255) // 256), 1, 1),
        )
        z1_biased = z1_biased_flat.reshape(M, N)

        a1 = np.zeros_like(z1_biased)
        total1 = M * N
        [a1_flat] = self._run(
            "relu_forward",
            [z1_biased.flatten(), a1.flatten()],
            [(total1, "u32")],
            [(1, total1)],
            workgroup=(min(256, total1), 1, 1),
            grid=(max(1, (total1 + 255) // 256), 1, 1),
        )
        a1 = a1_flat.reshape(M, N)

        M2, K2, N2 = batch_size, self.hidden_dim, self.output_dim
        z2 = np.zeros((M2, N2), dtype=np.float32)
        [z2_flat] = self._run(
            "matmul",
            [a1.flatten(), self.W2.flatten(), z2.flatten()],
            [(M2, "u32"), (K2, "u32"), (N2, "u32")],
            [(2, M2 * N2)],
            workgroup=(min(256, M2 * N2), 1, 1),
            grid=(max(1, (M2 * N2 + 255) // 256), 1, 1),
        )
        z2 = z2_flat.reshape(M2, N2)

        z2_biased = np.zeros_like(z2)
        [z2_biased_flat] = self._run(
            "bias_add",
            [z2.flatten(), self.b2, z2_biased.flatten()],
            [(M2, "u32"), (N2, "u32")],
            [(2, M2 * N2)],
            workgroup=(min(256, M2 * N2), 1, 1),
            grid=(max(1, (M2 * N2 + 255) // 256), 1, 1),
        )
        z2_biased = z2_biased_flat.reshape(M2, N2)

        softmax_out = np.zeros_like(z2_biased)
        [sm_flat] = self._run(
            "softmax_forward",
            [z2_biased.flatten(), softmax_out.flatten()],
            [(M2, "u32"), (N2, "u32")],
            [(1, M2 * N2)],
            workgroup=(min(256, M2), 1, 1),
            grid=(max(1, (M2 + 255) // 256), 1, 1),
        )
        softmax_out = sm_flat.reshape(M2, N2)

        cache = {
            "X": X_batch, "z1_biased": z1_biased, "a1": a1,
            "softmax_out": softmax_out, "batch_size": batch_size,
        }
        return softmax_out, cache

    def compute_loss(self, softmax_out, labels):
        batch_size = softmax_out.shape[0]
        cols = softmax_out.shape[1]
        labels_f32 = labels.astype(np.float32)
        losses = np.zeros(batch_size, dtype=np.float32)

        [loss_vals] = self._run(
            "cross_entropy_loss",
            [softmax_out.flatten(), labels_f32, losses],
            [(batch_size, "u32"), (cols, "u32")],
            [(2, batch_size)],
            workgroup=(min(256, batch_size), 1, 1),
            grid=(max(1, (batch_size + 255) // 256), 1, 1),
        )
        return float(np.mean(loss_vals))

    def backward(self, labels, cache):
        batch_size = cache["batch_size"]
        softmax_out = cache["softmax_out"]
        a1 = cache["a1"]
        z1_biased = cache["z1_biased"]
        X = cache["X"]
        cols = self.output_dim
        labels_f32 = labels.astype(np.float32)

        d_z2 = np.zeros_like(softmax_out)
        [d_z2_flat] = self._run(
            "softmax_ce_backward",
            [softmax_out.flatten(), labels_f32, d_z2.flatten()],
            [(batch_size, "u32"), (cols, "u32")],
            [(2, batch_size * cols)],
            workgroup=(min(256, batch_size), 1, 1),
            grid=(max(1, (batch_size + 255) // 256), 1, 1),
        )
        d_z2 = d_z2_flat.reshape(batch_size, cols)
        d_z2 = d_z2 / batch_size

        M_at, K_at, N_at = batch_size, self.hidden_dim, self.output_dim
        dW2 = np.zeros((self.hidden_dim, self.output_dim), dtype=np.float32)
        [dW2_flat] = self._run(
            "matmul_at_b",
            [a1.flatten(), d_z2.flatten(), dW2.flatten()],
            [(M_at, "u32"), (K_at, "u32"), (N_at, "u32")],
            [(2, K_at * N_at)],
            workgroup=(min(256, K_at * N_at), 1, 1),
            grid=(max(1, (K_at * N_at + 255) // 256), 1, 1),
        )
        dW2 = dW2_flat.reshape(self.hidden_dim, self.output_dim)

        db2 = np.zeros(self.output_dim, dtype=np.float32)
        [db2_vals] = self._run(
            "bias_grad",
            [d_z2.flatten(), db2],
            [(batch_size, "u32"), (self.output_dim, "u32")],
            [(1, self.output_dim)],
            workgroup=(min(256, self.output_dim), 1, 1),
        )
        db2 = db2_vals

        M_abt, K_abt, N_abt = batch_size, self.hidden_dim, self.output_dim
        d_a1 = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        [d_a1_flat] = self._run(
            "matmul_a_bt",
            [d_z2.flatten(), self.W2.flatten(), d_a1.flatten()],
            [(M_abt, "u32"), (K_abt, "u32"), (N_abt, "u32")],
            [(2, M_abt * K_abt)],
            workgroup=(min(256, M_abt * K_abt), 1, 1),
            grid=(max(1, (M_abt * K_abt + 255) // 256), 1, 1),
        )
        d_a1 = d_a1_flat.reshape(batch_size, self.hidden_dim)

        total_hidden = batch_size * self.hidden_dim
        d_z1 = np.zeros_like(d_a1)
        [d_z1_flat] = self._run(
            "relu_backward",
            [d_a1.flatten(), z1_biased.flatten(), d_z1.flatten()],
            [(total_hidden, "u32")],
            [(2, total_hidden)],
            workgroup=(min(256, total_hidden), 1, 1),
            grid=(max(1, (total_hidden + 255) // 256), 1, 1),
        )
        d_z1 = d_z1_flat.reshape(batch_size, self.hidden_dim)

        M_at1, K_at1, N_at1 = batch_size, self.input_dim, self.hidden_dim
        dW1 = np.zeros((self.input_dim, self.hidden_dim), dtype=np.float32)
        [dW1_flat] = self._run(
            "matmul_at_b",
            [X.flatten(), d_z1.flatten(), dW1.flatten()],
            [(M_at1, "u32"), (K_at1, "u32"), (N_at1, "u32")],
            [(2, K_at1 * N_at1)],
            workgroup=(min(256, K_at1 * N_at1), 1, 1),
            grid=(max(1, (K_at1 * N_at1 + 255) // 256), 1, 1),
        )
        dW1 = dW1_flat.reshape(self.input_dim, self.hidden_dim)

        db1 = np.zeros(self.hidden_dim, dtype=np.float32)
        [db1_vals] = self._run(
            "bias_grad",
            [d_z1.flatten(), db1],
            [(batch_size, "u32"), (self.hidden_dim, "u32")],
            [(1, self.hidden_dim)],
            workgroup=(min(256, self.hidden_dim), 1, 1),
        )
        db1 = db1_vals

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        lr_buf = np.array([self.lr], dtype=np.float32)

        for param, grad in [
            (self.W1, dW1), (self.b1, db1),
            (self.W2, dW2), (self.b2, db2),
        ]:
            n = param.size
            [updated] = self._run(
                "sgd_update",
                [param.flatten(), grad.flatten(), lr_buf],
                [(n, "u32")],
                [(0, n)],
                workgroup=(min(256, n), 1, 1),
                grid=(max(1, (n + 255) // 256), 1, 1),
            )
            np.copyto(param.reshape(-1), updated)

    def predict(self, X_batch):
        softmax_out, _ = self.forward(X_batch)
        return np.argmax(softmax_out, axis=1)

    def evaluate(self, X, y, batch_size=32):
        correct = 0
        total = 0
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            preds = self.predict(X_batch)
            correct += np.sum(preds == y_batch)
            total += len(y_batch)
        return correct / total


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
    shuffle_seed=7,
):
    data_dir = os.path.join(WAVE_ROOT, "data")
    print("Loading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    print(f"\nInitializing network: 784 -> {hidden_dim} -> 10")
    print(f"  Backend: Apple Metal GPU (via WAVE)")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Shuffle seed: {shuffle_seed}")

    model = WaveMetalNN(input_dim=784, hidden_dim=hidden_dim, output_dim=10, lr=lr)

    print("\nCompiled WAVE kernels:")
    for name in model.kernels:
        print(f"  {name}")

    epoch_indices = generate_batch_indices(len(train_images), epochs, seed=shuffle_seed)

    print("\nTraining started (Apple Metal GPU)")
    results = []
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        indices = epoch_indices[epoch]
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx in range(0, len(train_images), batch_size):
            if max_batches is not None and n_batches >= max_batches:
                break

            batch_indices = indices[batch_idx:batch_idx + batch_size]
            if len(batch_indices) < batch_size:
                continue

            X_batch = train_images[batch_indices]
            y_batch = train_labels[batch_indices]

            softmax_out, cache = model.forward(X_batch)
            loss = model.compute_loss(softmax_out, y_batch)
            dW1, db1, dW2, db2 = model.backward(y_batch, cache)
            model.update_weights(dW1, db1, dW2, db2)

            epoch_loss += loss
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

        test_acc = model.evaluate(test_images, test_labels, batch_size=batch_size)
        train_acc = model.evaluate(train_images[:1000], train_labels[:1000], batch_size=batch_size)

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
    print(f"\nTraining complete (Apple Metal GPU)")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final test accuracy: {results[-1]['test_acc']:.4f}")
    print(f"Final train accuracy: {results[-1]['train_acc']:.4f}")

    return model, results


if __name__ == "__main__":
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_batches = int(sys.argv[3]) if len(sys.argv) > 3 else None
    shuffle_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7

    model, results = train(
        epochs=epochs,
        batch_size=batch_size,
        lr=0.1,
        hidden_dim=128,
        eval_every=50,
        max_batches=max_batches,
        shuffle_seed=shuffle_seed,
    )
