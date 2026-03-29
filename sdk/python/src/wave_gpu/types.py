# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Type annotations for WAVE GPU kernel parameters."""

from typing import Any


class _TypeDescriptor:
    """Base class for WAVE type annotations used in kernel signatures."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __getitem__(self, _key: Any) -> "_TypeDescriptor":
        return _TypeDescriptor(f"{self._name}[]")


f32 = _TypeDescriptor("f32")
f64 = _TypeDescriptor("f64")
f16 = _TypeDescriptor("f16")
u32 = _TypeDescriptor("u32")
i32 = _TypeDescriptor("i32")
