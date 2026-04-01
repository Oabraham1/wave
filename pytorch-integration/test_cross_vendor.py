# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from wave_kernels import ALL_KERNELS
from wave_runtime import compile_kernel, run_kernel
from wave_metal_runner import run_kernel_metal


def float_to_hex(f):
    return struct.pack("<f", f).hex()


def compare(name, emu_result, metal_result, rtol=1e-4, atol=1e-5):
    match = np.allclose(emu_result, metal_result, rtol=rtol, atol=atol)
    bit_exact = np.array_equal(
        emu_result.view(np.uint32), metal_result.view(np.uint32)
    )
    max_diff = np.max(np.abs(emu_result - metal_result))
    max_rel = np.max(
        np.abs(emu_result - metal_result)
        / np.maximum(np.abs(emu_result), 1e-30)
    )

    status = "BIT-EXACT" if bit_exact else ("MATCH" if match else "MISMATCH")
    print(f"[{status}] {name}")
    if not bit_exact:
        print(f"  Max abs diff:  {max_diff:.2e}")
        print(f"  Max rel diff:  {max_rel:.2e}")
        diff_mask = emu_result.view(np.uint32) != metal_result.view(np.uint32)
        if diff_mask.any():
            idx = np.argmax(diff_mask)
            print(f"  First diff at [{idx}]: emu={emu_result[idx]:.8e} ({float_to_hex(emu_result[idx])}) "
                  f"metal={metal_result[idx]:.8e} ({float_to_hex(metal_result[idx])})")
    return status


def run_both(wbin, buffers, scalars, output_specs, workgroup):
    emu = run_kernel(wbin, buffers, scalars, output_specs, workgroup=workgroup)
    metal = run_kernel_metal(wbin, buffers, scalars, output_specs, workgroup=workgroup)
    return emu, metal


results = {"BIT-EXACT": 0, "MATCH": 0, "MISMATCH": 0}

print("Cross-Vendor Verification: WAVE Emulator vs Apple M2 Metal GPU")

wbin = compile_kernel("matmul", ALL_KERNELS["matmul"])
A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
C = np.zeros((2, 2), dtype=np.float32)
M, K, N = 2, 3, 2
emu, metal = run_both(wbin, [A.flatten(), B.flatten(), C.flatten()],
                      [(M, "u32"), (K, "u32"), (N, "u32")], [(2, M*N)], (M*N, 1, 1))
s = compare("matmul", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("bias_add", ALL_KERNELS["bias_add"])
X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
bias = np.array([10, 20, 30], dtype=np.float32)
out = np.zeros((2, 3), dtype=np.float32)
emu, metal = run_both(wbin, [X.flatten(), bias, out.flatten()],
                      [(2, "u32"), (3, "u32")], [(2, 6)], (6, 1, 1))
s = compare("bias_add", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("relu_forward", ALL_KERNELS["relu_forward"])
x = np.array([-3, -1, 0, 0.5, 2, -0.1, 7, -100], dtype=np.float32)
y = np.zeros_like(x)
emu, metal = run_both(wbin, [x, y], [(8, "u32")], [(1, 8)], (8, 1, 1))
s = compare("relu_forward", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("relu_backward", ALL_KERNELS["relu_backward"])
grad_out = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
fwd_out = np.array([0, 0, 5, 0, 3, 0.1], dtype=np.float32)
grad_in = np.zeros_like(grad_out)
emu, metal = run_both(wbin, [grad_out, fwd_out, grad_in], [(6, "u32")], [(2, 6)], (6, 1, 1))
s = compare("relu_backward", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("softmax_forward", ALL_KERNELS["softmax_forward"])
logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
output = np.zeros_like(logits)
emu, metal = run_both(wbin, [logits.flatten(), output.flatten()],
                      [(2, "u32"), (3, "u32")], [(1, 6)], (256, 1, 1))
s = compare("softmax_forward", emu[0], metal[0], rtol=1e-3)
results[s] += 1

wbin = compile_kernel("cross_entropy_loss", ALL_KERNELS["cross_entropy_loss"])
softmax_out = np.array([[0.09, 0.24, 0.67], [0.7, 0.2, 0.1]], dtype=np.float32)
labels = np.array([2, 0], dtype=np.float32)
losses = np.zeros(2, dtype=np.float32)
emu, metal = run_both(wbin, [softmax_out.flatten(), labels, losses],
                      [(2, "u32"), (3, "u32")], [(2, 2)], (256, 1, 1))
s = compare("cross_entropy_loss", emu[0], metal[0], rtol=1e-2)
results[s] += 1

wbin = compile_kernel("softmax_ce_backward", ALL_KERNELS["softmax_ce_backward"])
grad = np.zeros_like(softmax_out)
emu, metal = run_both(wbin, [softmax_out.flatten(), labels, grad.flatten()],
                      [(2, "u32"), (3, "u32")], [(2, 6)], (256, 1, 1))
s = compare("softmax_ce_backward", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("matmul_at_b", ALL_KERNELS["matmul_at_b"])
A2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
B2 = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float32)
C2 = np.zeros((2, 3), dtype=np.float32)
emu, metal = run_both(wbin, [A2.flatten(), B2.flatten(), C2.flatten()],
                      [(3, "u32"), (2, "u32"), (3, "u32")], [(2, 6)], (6, 1, 1))
s = compare("matmul_at_b", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("matmul_a_bt", ALL_KERNELS["matmul_a_bt"])
A3 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
B3 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
C3 = np.zeros((2, 2), dtype=np.float32)
emu, metal = run_both(wbin, [A3.flatten(), B3.flatten(), C3.flatten()],
                      [(2, "u32"), (2, "u32"), (3, "u32")], [(2, 4)], (4, 1, 1))
s = compare("matmul_a_bt", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("bias_grad", ALL_KERNELS["bias_grad"])
grad_bg = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
bias_g = np.zeros(3, dtype=np.float32)
emu, metal = run_both(wbin, [grad_bg.flatten(), bias_g],
                      [(3, "u32"), (3, "u32")], [(1, 3)], (3, 1, 1))
s = compare("bias_grad", emu[0], metal[0])
results[s] += 1

wbin = compile_kernel("sgd_update", ALL_KERNELS["sgd_update"])
param = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
grad_sgd = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
lr_buf = np.array([0.01], dtype=np.float32)
emu, metal = run_both(wbin, [param, grad_sgd, lr_buf], [(4, "u32")], [(0, 4)], (4, 1, 1))
s = compare("sgd_update", emu[0], metal[0])
results[s] += 1

print()
print(f"Summary: {results['BIT-EXACT']} bit-exact, {results['MATCH']} close match, {results['MISMATCH']} mismatch")
print(f"Total: {sum(results.values())} kernels tested")
