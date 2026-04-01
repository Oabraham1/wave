# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""WAVE GPU SDK for Python.

Write GPU kernels in Python, run on any GPU. Supports Apple Metal,
NVIDIA CUDA, AMD ROCm, Intel SYCL, and a built-in emulator.
"""

from .array import WaveArray, array, ones, zeros
from .device import DeviceInfo, device
from .kernel import kernel
from .types import f16, f32, f64, i32, u32

__version__ = "0.1.2"

__all__ = [
    "WaveArray",
    "array",
    "ones",
    "zeros",
    "DeviceInfo",
    "device",
    "kernel",
    "f16",
    "f32",
    "f64",
    "i32",
    "u32",
]


def thread_id() -> int:
    """Placeholder for thread_id() intrinsic used in kernel source."""
    raise RuntimeError("thread_id() can only be called inside a @kernel function")


def workgroup_id() -> int:
    """Placeholder for workgroup_id() intrinsic used in kernel source."""
    raise RuntimeError("workgroup_id() can only be called inside a @kernel function")


def lane_id() -> int:
    """Placeholder for lane_id() intrinsic used in kernel source."""
    raise RuntimeError("lane_id() can only be called inside a @kernel function")


def wave_width() -> int:
    """Placeholder for wave_width() intrinsic used in kernel source."""
    raise RuntimeError("wave_width() can only be called inside a @kernel function")


def barrier() -> None:
    """Placeholder for barrier() intrinsic used in kernel source."""
    raise RuntimeError("barrier() can only be called inside a @kernel function")
