# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Array types for WAVE GPU kernel data."""

from typing import List, Sequence, Union


class WaveArray:
    """CPU-side array that can be passed to WAVE GPU kernels."""

    def __init__(self, data: Sequence[Union[int, float]], dtype: str = "f32") -> None:
        self.data: List[float] = [float(x) for x in data]
        self.dtype: str = dtype

    def to_list(self) -> List[float]:
        """Return the array contents as a Python list."""
        return list(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    def __repr__(self) -> str:
        return f"WaveArray({self.data}, dtype='{self.dtype}')"


def array(data: Sequence[Union[int, float]], dtype: str = "f32") -> WaveArray:
    """Create a WAVE array from a Python sequence."""
    return WaveArray(data, dtype)


def zeros(n: int, dtype: str = "f32") -> WaveArray:
    """Create a zero-filled WAVE array."""
    return WaveArray([0.0] * n, dtype)


def ones(n: int, dtype: str = "f32") -> WaveArray:
    """Create a WAVE array filled with ones."""
    return WaveArray([1.0] * n, dtype)
