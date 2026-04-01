# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from wave_kernels import ALL_KERNELS
from wave_runtime import compile_kernel, run_kernel

WAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def initialize_weights(seed=42):
    rng = np.random.RandomState(seed)
    input_dim, hidden_dim, output_dim = 784, 128, 10

    scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
    W1 = (rng.randn(input_dim, hidden_dim) * scale1).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)

    scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
    W2 = (rng.randn(hidden_dim, output_dim) * scale2).astype(np.float32)
    b2 = np.zeros(output_dim, dtype=np.float32)

    return W1, b1, W2, b2


def wave_forward_backward(X_batch, labels, W1, b1, W2, b2):
    kernels = {}
    for name, source in ALL_KERNELS.items():
        kernels[name] = compile_kernel(name, source)

    batch_size = X_batch.shape[0]
    input_dim, hidden_dim, output_dim = 784, 128, 10

    M, K, N = batch_size, input_dim, hidden_dim
    z1 = np.zeros((M, N), dtype=np.float32)
    [z1_flat] = run_kernel(kernels["matmul"],
        [X_batch.flatten(), W1.flatten(), z1.flatten()],
        [(M, "u32"), (K, "u32"), (N, "u32")],
        [(2, M * N)],
        workgroup=(min(256, M * N), 1, 1),
        grid=(max(1, (M * N + 255) // 256), 1, 1))
    z1 = z1_flat.reshape(M, N)

    z1b = np.zeros_like(z1)
    [z1b_flat] = run_kernel(kernels["bias_add"],
        [z1.flatten(), b1, z1b.flatten()],
        [(M, "u32"), (N, "u32")],
        [(2, M * N)],
        workgroup=(min(256, M * N), 1, 1),
        grid=(max(1, (M * N + 255) // 256), 1, 1))
    z1b = z1b_flat.reshape(M, N)

    a1 = np.zeros_like(z1b)
    total1 = M * N
    [a1_flat] = run_kernel(kernels["relu_forward"],
        [z1b.flatten(), a1.flatten()],
        [(total1, "u32")],
        [(1, total1)],
        workgroup=(min(256, total1), 1, 1),
        grid=(max(1, (total1 + 255) // 256), 1, 1))
    a1 = a1_flat.reshape(M, N)

    M2, K2, N2 = batch_size, hidden_dim, output_dim
    z2 = np.zeros((M2, N2), dtype=np.float32)
    [z2_flat] = run_kernel(kernels["matmul"],
        [a1.flatten(), W2.flatten(), z2.flatten()],
        [(M2, "u32"), (K2, "u32"), (N2, "u32")],
        [(2, M2 * N2)],
        workgroup=(min(256, M2 * N2), 1, 1),
        grid=(max(1, (M2 * N2 + 255) // 256), 1, 1))
    z2 = z2_flat.reshape(M2, N2)

    z2b = np.zeros_like(z2)
    [z2b_flat] = run_kernel(kernels["bias_add"],
        [z2.flatten(), b2, z2b.flatten()],
        [(M2, "u32"), (N2, "u32")],
        [(2, M2 * N2)],
        workgroup=(min(256, M2 * N2), 1, 1),
        grid=(max(1, (M2 * N2 + 255) // 256), 1, 1))
    z2b = z2b_flat.reshape(M2, N2)

    softmax_out = np.zeros_like(z2b)
    [sm_flat] = run_kernel(kernels["softmax_forward"],
        [z2b.flatten(), softmax_out.flatten()],
        [(M2, "u32"), (N2, "u32")],
        [(1, M2 * N2)],
        workgroup=(min(256, M2), 1, 1),
        grid=(max(1, (M2 + 255) // 256), 1, 1))
    softmax_out = sm_flat.reshape(M2, N2)

    labels_f32 = labels.astype(np.float32)
    cols = output_dim

    d_z2 = np.zeros_like(softmax_out)
    [d_z2_flat] = run_kernel(kernels["softmax_ce_backward"],
        [softmax_out.flatten(), labels_f32, d_z2.flatten()],
        [(batch_size, "u32"), (cols, "u32")],
        [(2, batch_size * cols)],
        workgroup=(min(256, batch_size), 1, 1),
        grid=(max(1, (batch_size + 255) // 256), 1, 1))
    d_z2 = d_z2_flat.reshape(batch_size, cols) / batch_size

    dW2 = np.zeros((hidden_dim, output_dim), dtype=np.float32)
    [dW2_flat] = run_kernel(kernels["matmul_at_b"],
        [a1.flatten(), d_z2.flatten(), dW2.flatten()],
        [(batch_size, "u32"), (hidden_dim, "u32"), (output_dim, "u32")],
        [(2, hidden_dim * output_dim)],
        workgroup=(min(256, hidden_dim * output_dim), 1, 1),
        grid=(max(1, (hidden_dim * output_dim + 255) // 256), 1, 1))
    dW2 = dW2_flat.reshape(hidden_dim, output_dim)

    db2 = np.zeros(output_dim, dtype=np.float32)
    [db2] = run_kernel(kernels["bias_grad"],
        [d_z2.flatten(), db2],
        [(batch_size, "u32"), (output_dim, "u32")],
        [(1, output_dim)],
        workgroup=(min(256, output_dim), 1, 1))

    d_a1 = np.zeros((batch_size, hidden_dim), dtype=np.float32)
    [d_a1_flat] = run_kernel(kernels["matmul_a_bt"],
        [d_z2.flatten(), W2.flatten(), d_a1.flatten()],
        [(batch_size, "u32"), (hidden_dim, "u32"), (output_dim, "u32")],
        [(2, batch_size * hidden_dim)],
        workgroup=(min(256, batch_size * hidden_dim), 1, 1),
        grid=(max(1, (batch_size * hidden_dim + 255) // 256), 1, 1))
    d_a1 = d_a1_flat.reshape(batch_size, hidden_dim)

    total_hidden = batch_size * hidden_dim
    d_z1 = np.zeros_like(d_a1)
    [d_z1_flat] = run_kernel(kernels["relu_backward"],
        [d_a1.flatten(), z1b.flatten(), d_z1.flatten()],
        [(total_hidden, "u32")],
        [(2, total_hidden)],
        workgroup=(min(256, total_hidden), 1, 1),
        grid=(max(1, (total_hidden + 255) // 256), 1, 1))
    d_z1 = d_z1_flat.reshape(batch_size, hidden_dim)

    dW1 = np.zeros((input_dim, hidden_dim), dtype=np.float32)
    [dW1_flat] = run_kernel(kernels["matmul_at_b"],
        [X_batch.flatten(), d_z1.flatten(), dW1.flatten()],
        [(batch_size, "u32"), (input_dim, "u32"), (hidden_dim, "u32")],
        [(2, input_dim * hidden_dim)],
        workgroup=(min(256, input_dim * hidden_dim), 1, 1),
        grid=(max(1, (input_dim * hidden_dim + 255) // 256), 1, 1))
    dW1 = dW1_flat.reshape(input_dim, hidden_dim)

    db1 = np.zeros(hidden_dim, dtype=np.float32)
    [db1] = run_kernel(kernels["bias_grad"],
        [d_z1.flatten(), db1],
        [(batch_size, "u32"), (hidden_dim, "u32")],
        [(1, hidden_dim)],
        workgroup=(min(256, hidden_dim), 1, 1))

    return dW1, db1, dW2, db2


def pytorch_forward_backward(X_batch, labels, W1, b1, W2, b2):
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(W1.T))
        model[0].bias.copy_(torch.from_numpy(b1))
        model[2].weight.copy_(torch.from_numpy(W2.T))
        model[2].bias.copy_(torch.from_numpy(b2))

    X_t = torch.from_numpy(X_batch)
    y_t = torch.from_numpy(labels)

    output = model(X_t)
    loss = nn.CrossEntropyLoss()(output, y_t)
    loss.backward()

    dW1_pt = model[0].weight.grad.numpy().T.copy()
    db1_pt = model[0].bias.grad.numpy().copy()
    dW2_pt = model[2].weight.grad.numpy().T.copy()
    db2_pt = model[2].bias.grad.numpy().copy()

    return dW1_pt, db1_pt, dW2_pt, db2_pt, loss.item()


def compare_grads(name, wave_grad, pt_grad):
    max_abs_diff = np.max(np.abs(wave_grad - pt_grad))
    mean_abs = np.mean(np.abs(pt_grad))
    max_rel_diff = np.max(np.abs(wave_grad - pt_grad) / np.maximum(np.abs(pt_grad), 1e-30))
    close = np.allclose(wave_grad, pt_grad, rtol=1e-4, atol=1e-6)
    bit_exact = np.array_equal(wave_grad.view(np.uint32), pt_grad.view(np.uint32))

    status = "BIT-EXACT" if bit_exact else ("CLOSE" if close else "DIFFERS")
    print(f"  [{status}] {name}")
    print(f"    Shape: {wave_grad.shape}, Mean |grad|: {mean_abs:.6e}")
    print(f"    Max abs diff: {max_abs_diff:.6e}, Max rel diff: {max_rel_diff:.6e}")
    if not close:
        diff_mask = np.abs(wave_grad - pt_grad) > 1e-6
        n_diff = np.sum(diff_mask)
        print(f"    Elements differing (>1e-6): {n_diff}/{wave_grad.size}")
    return status


if __name__ == "__main__":
    from torchvision import datasets, transforms

    data_dir = os.path.join(WAVE_ROOT, "data")
    train_ds = datasets.MNIST(data_dir, train=True, download=False,
                              transform=transforms.ToTensor())
    train_images = np.array([img.numpy().flatten() for img, _ in train_ds], dtype=np.float32)
    train_labels = np.array([label for _, label in train_ds], dtype=np.int64)

    W1, b1, W2, b2 = initialize_weights(seed=42)

    X_batch = train_images[:32]
    y_batch = train_labels[:32]

    print("Gradient Comparison: WAVE Emulator vs PyTorch")
    print(f"Batch: first 32 training samples")
    print(f"Weights: Xavier init (seed=42)")

    print("Running PyTorch forward+backward...")
    dW1_pt, db1_pt, dW2_pt, db2_pt, pt_loss = pytorch_forward_backward(
        X_batch, y_batch, W1.copy(), b1.copy(), W2.copy(), b2.copy()
    )
    print(f"  PyTorch loss: {pt_loss:.8f}")

    print("\nRunning WAVE forward+backward...")
    dW1_wave, db1_wave, dW2_wave, db2_wave = wave_forward_backward(
        X_batch, y_batch, W1.copy(), b1.copy(), W2.copy(), b2.copy()
    )

    print("\nGradient comparison:")
    results = []
    results.append(compare_grads("dW1 (784x128)", dW1_wave, dW1_pt))
    results.append(compare_grads("db1 (128,)", db1_wave, db1_pt))
    results.append(compare_grads("dW2 (128x10)", dW2_wave, dW2_pt))
    results.append(compare_grads("db2 (10,)", db2_wave, db2_pt))

    print(f"\nSummary: {results.count('BIT-EXACT')} bit-exact, "
          f"{results.count('CLOSE')} close, "
          f"{results.count('DIFFERS')} differs")
