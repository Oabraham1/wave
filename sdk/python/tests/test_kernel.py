# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel compilation and the @kernel decorator."""

import pytest

import wave_gpu


def test_kernel_decorator():
    @wave_gpu.kernel
    def vector_add(a, b, out, n):
        gid = wave_gpu.thread_id()
        if gid < n:
            out[gid] = a[gid] + b[gid]

    assert hasattr(vector_add, "_source")
    assert "vector_add" in vector_add._source


def test_thread_id_outside_kernel():
    with pytest.raises(RuntimeError, match="inside a @kernel"):
        wave_gpu.thread_id()


def test_barrier_outside_kernel():
    with pytest.raises(RuntimeError, match="inside a @kernel"):
        wave_gpu.barrier()


def test_kernel_bad_arg_type():
    @wave_gpu.kernel
    def noop(x):
        pass

    with pytest.raises(TypeError, match="Unsupported argument type"):
        noop("not a valid arg")
