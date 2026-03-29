# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPU device detection."""

import wave_gpu


def test_device_returns_info():
    dev = wave_gpu.device()
    assert dev.vendor in ("apple", "nvidia", "amd", "intel", "emulator")
    assert len(dev.name) > 0


def test_device_repr():
    dev = wave_gpu.device()
    assert len(repr(dev)) > 0
