# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Kernel decorator and compilation for the WAVE Python SDK."""

from __future__ import annotations

import inspect
import textwrap
from typing import Any, Callable

from .array import WaveArray
from .runtime import CompiledKernel


class KernelWrapper:
    """Wraps a Python function as a WAVE GPU kernel."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self._source = textwrap.dedent(inspect.getsource(func))
        self._compiled: CompiledKernel | None = None

    def _ensure_compiled(self) -> CompiledKernel:
        if self._compiled is None:
            self._compiled = CompiledKernel(self._source, "python")
        return self._compiled

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Launch the kernel with the given arguments.

        WaveArray arguments are passed as device buffers.
        Integer arguments are passed as kernel parameters.
        """
        grid = kwargs.get("grid", (1, 1, 1))
        workgroup = kwargs.get("workgroup", (256, 1, 1))

        buffers: list[WaveArray] = []
        scalars: list[int] = []

        for arg in args:
            if isinstance(arg, WaveArray):
                buffers.append(arg)
            elif isinstance(arg, (int, float)):
                scalars.append(int(arg))
            else:
                raise TypeError(
                    f"Unsupported argument type: {type(arg).__name__}. "
                    "Expected WaveArray or int."
                )

        compiled = self._ensure_compiled()

        n_threads = max(len(b) for b in buffers) if buffers else 1
        if grid == (1, 1, 1) and workgroup == (256, 1, 1):
            wg_size = min(256, n_threads)
            n_groups = (n_threads + wg_size - 1) // wg_size
            grid = (n_groups, 1, 1)
            workgroup = (wg_size, 1, 1)

        compiled.launch(buffers, scalars, grid, workgroup)


def kernel(func: Callable[..., Any]) -> KernelWrapper:
    """Decorator that marks a Python function as a WAVE GPU kernel."""
    return KernelWrapper(func)
