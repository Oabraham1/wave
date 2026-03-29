# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Tests for the WaveArray type."""

import wave_gpu


def test_array_creation():
    a = wave_gpu.array([1.0, 2.0, 3.0])
    assert len(a) == 3
    assert a[0] == 1.0
    assert a[1] == 2.0
    assert a[2] == 3.0


def test_array_to_list():
    a = wave_gpu.array([1.0, 2.0, 3.0, 4.0])
    assert a.to_list() == [1.0, 2.0, 3.0, 4.0]


def test_zeros():
    a = wave_gpu.zeros(5)
    assert len(a) == 5
    assert a.to_list() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_ones():
    a = wave_gpu.ones(3)
    assert len(a) == 3
    assert a.to_list() == [1.0, 1.0, 1.0]


def test_array_dtype():
    a = wave_gpu.array([1, 2, 3], dtype="u32")
    assert a.dtype == "u32"
    assert a.to_list() == [1.0, 2.0, 3.0]


def test_array_from_integers():
    a = wave_gpu.array([1, 2, 3])
    assert a.to_list() == [1.0, 2.0, 3.0]


def test_array_repr():
    a = wave_gpu.array([1.0, 2.0])
    assert "WaveArray" in repr(a)
