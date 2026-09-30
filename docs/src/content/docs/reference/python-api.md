---
title: "Python API reference"
description: "API reference for the wave_gpu Python module."
---

The `wave_gpu` module provides Python bindings for WAVE GPU compute. It exposes array creation, kernel compilation, device intrinsics, and a kernel decorator for writing GPU kernels in Python.

## Installation

```bash
pip install wave-gpu
```

## Array creation

### `wave_gpu.array(data, dtype="f32") -> WaveArray`

Create a host array from a Python sequence.

```python
import wave_gpu

a = wave_gpu.array([1.0, 2.0, 3.0])
b = wave_gpu.array([1, 2, 3, 4], dtype="u32")
```

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `list` or iterable | Required | Source values, converted to Python floats. |
| `dtype` | `str` | `"f32"` | Element type. One of: `"f16"`, `"f32"`, `"f64"`, `"i32"`, `"u32"`. |

Returns: `WaveArray` containing a copy of the input values.

Elements must be convertible to Python floats. The constructor stores the dtype string without validating it.

---

### `wave_gpu.zeros(n, dtype="f32") -> WaveArray`

Create a host array of `n` zeros.

```python
buf = wave_gpu.zeros(1024)
buf_int = wave_gpu.zeros(256, dtype="i32")
```

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | Required | Number of elements. |
| `dtype` | `str` | `"f32"` | Element type. |

Returns: `WaveArray`

---

### `wave_gpu.ones(n, dtype="f32") -> WaveArray`

Create a host array of `n` ones.

```python
buf = wave_gpu.ones(512, dtype="f64")
```

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | Required | Number of elements. |
| `dtype` | `str` | `"f32"` | Element type. |

Returns: `WaveArray`

---

## WaveArray

`WaveArray` stores host values used as kernel buffers. Array factories return this type; dispatch updates the supplied arrays.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | `list[float]` | Stored host values. |
| `dtype` | `str` | Element type string; such as `"f32"`, `"u32"`. |

### Methods

#### `to_list() -> list`

Return a copy of the stored values.

```python
a = wave_gpu.array([1.0, 2.0, 3.0])
print(a.to_list())  # [1.0, 2.0, 3.0]
```

#### `__len__() -> int`

Return the number of elements in the buffer.

```python
a = wave_gpu.zeros(128)
len(a)  # 128
```

#### `__getitem__(idx) -> float | list[float]`

Access a stored value by index.

```python
a = wave_gpu.array([10.0, 20.0, 30.0])
a[1]  # 20.0
```

Negative indexing and slicing follow Python list behavior.

---

## Kernel decorator

### `@wave_gpu.kernel`

Decorator that marks a Python function as a GPU kernel. The decorator captures the source. Compilation occurs on the first call and the binary is cached.

```python
from wave_gpu import thread_id

@wave_gpu.kernel
def vector_add(a: wave_gpu.f32[:], b: wave_gpu.f32[:], out: wave_gpu.f32[:]):
    tid = thread_id()
    out[tid] = a[tid] + b[tid]
```

The callable accepts buffers, numeric scalar arguments, and dispatch parameters. It detects the device and supports emulator dispatch. GPU devices raise `NotImplementedError`; use [explicit emulator execution](/getting-started/quickstart/) or the [Rust SDK](/guides/rust-sdk/) for GPU dispatch.

On the emulator:

```python
a = wave_gpu.array([1.0, 2.0, 3.0, 4.0])
b = wave_gpu.array([5.0, 6.0, 7.0, 8.0])
out = wave_gpu.zeros(4)

vector_add(a, b, out, grid=(1, 1, 1), workgroup=(4, 1, 1))
print(out.to_list())  # [6.0, 8.0, 10.0, 12.0]
```

Dispatch parameters, keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `tuple[int, int, int]` | Inferred | Workgroup counts `[x, y, z]`. |
| `workgroup` | `tuple[int, int, int]` | Inferred | Threads per workgroup `[x, y, z]`. |

---

## Device query

### `wave_gpu.device() -> DeviceInfo`

Return the detected device, with an emulator fallback when no GPU is found.

```python
dev = wave_gpu.device()
print(dev.vendor)  # "apple", "nvidia", "amd", "intel", or "emulator"
print(dev.name)    # e.g. "AMD Radeon RX 7900 XTX"
```

Returns: `DeviceInfo` with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `vendor` | `str` | GPU vendor string. |
| `name` | `str` | GPU device name. |

---

## Kernel intrinsics

The following intrinsics are available inside `@wave_gpu.kernel` functions. They must not be called outside of a kernel context.

### `wave_gpu.thread_id() -> int`

Returns the global thread index of the current invocation.

### `wave_gpu.workgroup_id() -> int`

Returns the workgroup index of the current invocation.

### `wave_gpu.lane_id() -> int`

Returns the lane index within the current wave, 0 to `wave_width() - 1`.

### `wave_gpu.wave_width() -> int`

Return the number of lanes per wave, typically 32 or 64.

### `wave_gpu.barrier()`

Synchronize all threads in the current workgroup. All threads must reach the barrier before any thread proceeds past it. Must not be called inside divergent control flow.

---

## Types

Type annotations used in kernel function signatures to declare buffer element types.

| Type | Description | Size |
|------|-------------|------|
| `wave_gpu.f16` | 16-bit IEEE 754 half-precision float | 2 bytes |
| `wave_gpu.f32` | 32-bit IEEE 754 single-precision float | 4 bytes |
| `wave_gpu.f64` | 64-bit IEEE 754 double-precision float | 8 bytes |
| `wave_gpu.i32` | 32-bit signed integer | 4 bytes |
| `wave_gpu.u32` | 32-bit unsigned integer | 4 bytes |

These types are also valid values for the `dtype` string parameter: `"f16"`, `"f32"`, `"f64"`, `"i32"`, `"u32"`.
