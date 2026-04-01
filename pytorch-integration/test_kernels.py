# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from wave_kernels import ALL_KERNELS
from wave_runtime import compile_kernel, run_kernel

PASS = 0
FAIL = 0


def check(name, actual, expected, rtol=1e-4, atol=1e-5):
    global PASS, FAIL
    if np.allclose(actual, expected, rtol=rtol, atol=atol):
        print(f"[PASS] {name}")
        PASS += 1
    else:
        print(f"[FAIL] {name}")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        print(f"  Max diff: {np.max(np.abs(actual - expected))}")
        FAIL += 1


def test_matmul():
    wbin = compile_kernel("matmul", ALL_KERNELS["matmul"])
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    C = np.zeros((2, 2), dtype=np.float32)
    M, K, N = 2, 3, 2

    results = run_kernel(
        wbin,
        buffers=[A.flatten(), B.flatten(), C.flatten()],
        scalars=[(M, "u32"), (K, "u32"), (N, "u32")],
        output_specs=[(2, M * N)],
        workgroup=(M * N, 1, 1),
    )
    check("matmul 2x3 @ 3x2", results[0], (A @ B).flatten())


def test_bias_add():
    wbin = compile_kernel("bias_add", ALL_KERNELS["bias_add"])
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bias = np.array([10, 20, 30], dtype=np.float32)
    out = np.zeros((2, 3), dtype=np.float32)
    rows, cols = 2, 3

    results = run_kernel(
        wbin,
        buffers=[X.flatten(), bias, out.flatten()],
        scalars=[(rows, "u32"), (cols, "u32")],
        output_specs=[(2, rows * cols)],
        workgroup=(rows * cols, 1, 1),
    )
    expected = (X + bias).flatten()
    check("bias_add", results[0], expected)


def test_relu_forward():
    wbin = compile_kernel("relu_forward", ALL_KERNELS["relu_forward"])
    x = np.array([-3, -1, 0, 0.5, 2, -0.1, 7, -100], dtype=np.float32)
    y = np.zeros_like(x)
    n = len(x)

    results = run_kernel(
        wbin,
        buffers=[x, y],
        scalars=[(n, "u32")],
        output_specs=[(1, n)],
        workgroup=(n, 1, 1),
    )
    expected = np.maximum(x, 0)
    check("relu_forward", results[0], expected)


def test_relu_backward():
    wbin = compile_kernel("relu_backward", ALL_KERNELS["relu_backward"])
    grad_out = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    fwd_out = np.array([0, 0, 5, 0, 3, 0.1], dtype=np.float32)
    grad_in = np.zeros_like(grad_out)
    n = len(grad_out)

    results = run_kernel(
        wbin,
        buffers=[grad_out, fwd_out, grad_in],
        scalars=[(n, "u32")],
        output_specs=[(2, n)],
        workgroup=(n, 1, 1),
    )
    expected = grad_out * (fwd_out > 0).astype(np.float32)
    check("relu_backward", results[0], expected)


def test_softmax_forward():
    wbin = compile_kernel("softmax_forward", ALL_KERNELS["softmax_forward"])
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    output = np.zeros_like(logits)
    rows, cols = 2, 3

    results = run_kernel(
        wbin,
        buffers=[logits.flatten(), output.flatten()],
        scalars=[(rows, "u32"), (cols, "u32")],
        output_specs=[(1, rows * cols)],
        workgroup=(256, 1, 1),
    )
    result = results[0].reshape(rows, cols)
    exp_shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    check("softmax_forward", result.flatten(), expected.flatten(), rtol=1e-3)


def test_cross_entropy_loss():
    wbin = compile_kernel("cross_entropy_loss", ALL_KERNELS["cross_entropy_loss"])
    softmax_out = np.array([
        [0.09, 0.24, 0.67],
        [0.7, 0.2, 0.1]
    ], dtype=np.float32)
    labels = np.array([2, 0], dtype=np.float32)
    losses = np.zeros(2, dtype=np.float32)
    rows, cols = 2, 3

    results = run_kernel(
        wbin,
        buffers=[softmax_out.flatten(), labels, losses],
        scalars=[(rows, "u32"), (cols, "u32")],
        output_specs=[(2, 2)],
        workgroup=(256, 1, 1),
    )
    expected = np.array([
        -np.log(softmax_out[0, 2]),
        -np.log(softmax_out[1, 0])
    ], dtype=np.float32)
    check("cross_entropy_loss", results[0], expected, rtol=1e-2)


def test_softmax_ce_backward():
    wbin = compile_kernel("softmax_ce_backward", ALL_KERNELS["softmax_ce_backward"])
    softmax_out = np.array([
        [0.09, 0.24, 0.67],
        [0.7, 0.2, 0.1]
    ], dtype=np.float32)
    labels = np.array([2, 0], dtype=np.float32)
    grad = np.zeros_like(softmax_out)
    rows, cols = 2, 3

    results = run_kernel(
        wbin,
        buffers=[softmax_out.flatten(), labels, grad.flatten()],
        scalars=[(rows, "u32"), (cols, "u32")],
        output_specs=[(2, 6)],
        workgroup=(256, 1, 1),
    )
    expected = softmax_out.copy()
    expected[0, 2] -= 1.0
    expected[1, 0] -= 1.0
    check("softmax_ce_backward", results[0], expected.flatten())


def test_matmul_at_b():
    wbin = compile_kernel("matmul_at_b", ALL_KERNELS["matmul_at_b"])
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    B = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float32)
    M, K, N = 3, 2, 3
    C = np.zeros((K, N), dtype=np.float32)

    results = run_kernel(
        wbin,
        buffers=[A.flatten(), B.flatten(), C.flatten()],
        scalars=[(M, "u32"), (K, "u32"), (N, "u32")],
        output_specs=[(2, K * N)],
        workgroup=(K * N, 1, 1),
    )
    expected = A.T @ B
    check("matmul_at_b (A^T @ B)", results[0], expected.flatten())


def test_matmul_a_bt():
    wbin = compile_kernel("matmul_a_bt", ALL_KERNELS["matmul_a_bt"])
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
    M, K, N = 2, 2, 3
    C = np.zeros((M, K), dtype=np.float32)

    results = run_kernel(
        wbin,
        buffers=[A.flatten(), B.flatten(), C.flatten()],
        scalars=[(M, "u32"), (K, "u32"), (N, "u32")],
        output_specs=[(2, M * K)],
        workgroup=(M * K, 1, 1),
    )
    expected = A @ B.T
    check("matmul_a_bt (A @ B^T)", results[0], expected.flatten())


def test_bias_grad():
    wbin = compile_kernel("bias_grad", ALL_KERNELS["bias_grad"])
    grad = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    bias_g = np.zeros(3, dtype=np.float32)
    rows, cols = 3, 3

    results = run_kernel(
        wbin,
        buffers=[grad.flatten(), bias_g],
        scalars=[(rows, "u32"), (cols, "u32")],
        output_specs=[(1, cols)],
        workgroup=(cols, 1, 1),
    )
    expected = grad.sum(axis=0)
    check("bias_grad", results[0], expected)


def test_sgd_update():
    wbin = compile_kernel("sgd_update", ALL_KERNELS["sgd_update"])
    param = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    lr_buf = np.array([0.01], dtype=np.float32)
    n = 4

    results = run_kernel(
        wbin,
        buffers=[param, grad, lr_buf],
        scalars=[(n, "u32")],
        output_specs=[(0, n)],
        workgroup=(n, 1, 1),
    )
    expected = param - 0.01 * grad
    check("sgd_update", results[0], expected)


if __name__ == "__main__":
    test_matmul()
    test_bias_add()
    test_relu_forward()
    test_relu_backward()
    test_softmax_forward()
    test_cross_entropy_loss()
    test_softmax_ce_backward()
    test_matmul_at_b()
    test_matmul_a_bt()
    test_bias_grad()
    test_sgd_update()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
