#!/usr/bin/env python3
# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

# Loads a WAVE-compiled PTX kernel and executes it on an NVIDIA GPU via PyCUDA.
# Allocates a single device memory buffer, fills it with test data (two float arrays
# a and b), dispatches the kernel, and reads back results from array c.
#
# Usage: python run_on_gpu.py <program.ptx> [workgroup_count]
#
# Requires: pip install pycuda numpy
#

import sys
import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("Error: pycuda is required. Install with: pip install pycuda")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_on_gpu.py <program.ptx> [workgroup_count]")
        sys.exit(1)

    ptx_file = sys.argv[1]
    workgroup_count = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    with open(ptx_file, "r") as f:
        ptx_source = f.read()

    mod = cuda.module_from_buffer(ptx_source.encode("utf-8"))

    kernel_names = ["vector_add", "reduction", "wave_kernel"]
    kernel = None
    kernel_name = None
    for name in kernel_names:
        try:
            kernel = mod.get_function(name)
            kernel_name = name
            break
        except cuda.LogicError:
            continue

    if kernel is None:
        print("Error: no recognized kernel function found in PTX")
        sys.exit(1)

    print(f"Loaded kernel: {kernel_name}")

    element_count = 256 * workgroup_count
    buf = np.zeros(element_count * 3, dtype=np.float32)

    buf[:element_count] = np.arange(element_count, dtype=np.float32)
    buf[element_count : 2 * element_count] = (
        np.arange(element_count, dtype=np.float32) * 2.0
    )

    d_buf = cuda.mem_alloc(buf.nbytes)
    cuda.memcpy_htod(d_buf, buf)

    kernel(
        d_buf,
        block=(256, 1, 1),
        grid=(workgroup_count, 1, 1),
        shared=16384,
    )

    cuda.memcpy_dtoh(buf, d_buf)

    c = buf[2 * element_count :]
    print(f"Results (first 16 of {element_count} elements):")
    for i in range(min(16, element_count)):
        print(f"  c[{i}] = {c[i]}")

    expected = np.arange(min(16, element_count), dtype=np.float32) * 3.0
    actual = c[: min(16, element_count)]
    if np.allclose(actual, expected, atol=1e-6):
        print("\nVerification: PASSED")
    else:
        print("\nVerification: FAILED")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        sys.exit(1)


if __name__ == "__main__":
    main()
