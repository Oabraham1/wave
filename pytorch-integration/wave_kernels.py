# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

MATMUL_KERNEL = """
@kernel
def matmul(A: f32[:], B: f32[:], C: f32[:], M: u32, K: u32, N: u32):
    gid = workgroup_id() * 256 + thread_id()
    row = gid / N
    col = gid % N
    if row < M:
        if col < N:
            acc: f32 = 0.0
            for k in range(K):
                acc = acc + A[row * K + k] * B[k * N + col]
            C[row * N + col] = acc
"""

BIAS_ADD_KERNEL = """
@kernel
def bias_add(X: f32[:], bias: f32[:], out: f32[:], rows: u32, cols: u32):
    gid = workgroup_id() * 256 + thread_id()
    total = rows * cols
    if gid < total:
        col = gid % cols
        out[gid] = X[gid] + bias[col]
"""

RELU_FORWARD_KERNEL = """
@kernel
def relu_forward(x: f32[:], y: f32[:], n: u32):
    gid = workgroup_id() * 256 + thread_id()
    if gid < n:
        val = x[gid]
        if val > 0.0:
            y[gid] = val
        else:
            y[gid] = 0.0
"""

RELU_BACKWARD_KERNEL = """
@kernel
def relu_backward(grad_out: f32[:], fwd_out: f32[:], grad_in: f32[:], n: u32):
    gid = workgroup_id() * 256 + thread_id()
    if gid < n:
        if fwd_out[gid] > 0.0:
            grad_in[gid] = grad_out[gid]
        else:
            grad_in[gid] = 0.0
"""

SOFTMAX_FORWARD_KERNEL = """
@kernel
def softmax_forward(logits: f32[:], output: f32[:], rows: u32, cols: u32):
    row = workgroup_id() * 256 + thread_id()
    if row < rows:
        row_start = row * cols
        max_val: f32 = logits[row_start]
        for j in range(cols):
            cur = logits[row_start + j]
            if cur > max_val:
                max_val = cur
        sum_exp: f32 = 0.0
        for j in range(cols):
            diff = logits[row_start + j] - max_val
            exp_val = exp2(diff * 1.4426950)
            output[row_start + j] = exp_val
            sum_exp = sum_exp + exp_val
        for j in range(cols):
            output[row_start + j] = output[row_start + j] / sum_exp
"""

SOFTMAX_CROSS_ENTROPY_BACKWARD_KERNEL = """
@kernel
def softmax_ce_backward(softmax_out: f32[:], labels: f32[:], grad: f32[:], rows: u32, cols: u32):
    row = workgroup_id() * 256 + thread_id()
    if row < rows:
        row_start = row * cols
        label_idx = int(labels[row])
        for j in range(cols):
            grad[row_start + j] = softmax_out[row_start + j]
        grad[row_start + label_idx] = grad[row_start + label_idx] - 1.0
"""

CROSS_ENTROPY_LOSS_KERNEL = """
@kernel
def cross_entropy_loss(softmax_out: f32[:], labels: f32[:], losses: f32[:], rows: u32, cols: u32):
    row = workgroup_id() * 256 + thread_id()
    if row < rows:
        row_start = row * cols
        label_idx = int(labels[row])
        p = softmax_out[row_start + label_idx]
        neg_log_p = 0.0 - log2(p) * 0.6931472
        losses[row] = neg_log_p
"""

MATMUL_AT_B_KERNEL = """
@kernel
def matmul_at_b(A: f32[:], B: f32[:], C: f32[:], M: u32, K: u32, N: u32):
    gid = workgroup_id() * 256 + thread_id()
    row = gid / N
    col = gid % N
    if row < K:
        if col < N:
            acc: f32 = 0.0
            for m in range(M):
                acc = acc + A[m * K + row] * B[m * N + col]
            C[row * N + col] = acc
"""

MATMUL_A_BT_KERNEL = """
@kernel
def matmul_a_bt(A: f32[:], B: f32[:], C: f32[:], M: u32, K: u32, N: u32):
    gid = workgroup_id() * 256 + thread_id()
    row = gid / K
    col = gid % K
    if row < M:
        if col < K:
            acc: f32 = 0.0
            for n in range(N):
                acc = acc + A[row * N + n] * B[col * N + n]
            C[row * K + col] = acc
"""

BIAS_GRAD_KERNEL = """
@kernel
def bias_grad(grad: f32[:], bias_grad_out: f32[:], rows: u32, cols: u32):
    col = workgroup_id() * 256 + thread_id()
    if col < cols:
        acc: f32 = 0.0
        for r in range(rows):
            acc = acc + grad[r * cols + col]
        bias_grad_out[col] = acc
"""

SGD_UPDATE_KERNEL = """
@kernel
def sgd_update(param: f32[:], grad: f32[:], lr_buf: f32[:], n: u32):
    gid = workgroup_id() * 256 + thread_id()
    if gid < n:
        lr = lr_buf[0]
        param[gid] = param[gid] - lr * grad[gid]
"""

ALL_KERNELS = {
    "matmul": MATMUL_KERNEL,
    "bias_add": BIAS_ADD_KERNEL,
    "relu_forward": RELU_FORWARD_KERNEL,
    "relu_backward": RELU_BACKWARD_KERNEL,
    "softmax_forward": SOFTMAX_FORWARD_KERNEL,
    "softmax_ce_backward": SOFTMAX_CROSS_ENTROPY_BACKWARD_KERNEL,
    "cross_entropy_loss": CROSS_ENTROPY_LOSS_KERNEL,
    "matmul_at_b": MATMUL_AT_B_KERNEL,
    "matmul_a_bt": MATMUL_A_BT_KERNEL,
    "bias_grad": BIAS_GRAD_KERNEL,
    "sgd_update": SGD_UPDATE_KERNEL,
}
