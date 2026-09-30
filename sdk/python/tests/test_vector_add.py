# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute vector addition through the Python SDK emulator API."""

import wave_gpu
from wave_gpu.runtime import compile_kernel, launch_emulator


def test_vector_add_e2e():
    source = """
@kernel
def vector_add(a: f32[:], b: f32[:], out: f32[:], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"""
    a = wave_gpu.array([1.0, -2.5, 3.25, 4.0])
    b = wave_gpu.array([5.0, 6.5, -7.25, 8.0])
    out = wave_gpu.zeros(4)
    binary = compile_kernel(source, "python")
    launch_emulator(binary, [a, b, out], [4], (1, 1, 1), (4, 1, 1))
    assert out.to_list() == [6.0, 4.0, -4.0, 12.0]
    assert a.to_list() == [1.0, -2.5, 3.25, 4.0]
    assert b.to_list() == [5.0, 6.5, -7.25, 8.0]
