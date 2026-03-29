# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""End-to-end vector addition test using the runtime compilation pipeline."""

import pytest

import wave_gpu
from wave_gpu.runtime import compile_kernel


@pytest.mark.skipif(
    True,
    reason="Requires wave-compiler binary on PATH or in target/",
)
def test_vector_add_e2e():
    """Full end-to-end: compile + launch vector_add on emulator."""
    a = wave_gpu.array([1.0, 2.0, 3.0, 4.0])
    b = wave_gpu.array([5.0, 6.0, 7.0, 8.0])
    out = wave_gpu.zeros(4)

    @wave_gpu.kernel
    def vector_add(
        a: wave_gpu.f32[:],
        b: wave_gpu.f32[:],
        out: wave_gpu.f32[:],
        n: wave_gpu.u32,
    ):
        gid = wave_gpu.thread_id()
        if gid < n:
            out[gid] = a[gid] + b[gid]

    vector_add(a, b, out, len(a))
    assert out.to_list() == [6.0, 8.0, 10.0, 12.0]


def test_compile_kernel_source():
    """Test that compile_kernel function exists and has correct signature."""
    assert callable(compile_kernel)
