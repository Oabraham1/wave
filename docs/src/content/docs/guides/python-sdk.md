---
title: Python SDK
description: Getting started with the WAVE Python SDK (wave_gpu) for portable GPU programming.
---

The `wave_gpu` Python package lets you write GPU kernels in Python and run them on any supported GPU backend. This guide walks through installation, device setup, array management, kernel authoring, and error handling.

## Installation

Install from PyPI:

```bash
pip install wave-gpu
```

Import the package in your code:

```python
import wave_gpu
```

## Device Detection

Before launching kernels you need to confirm that a GPU is available. `wave_gpu.device()` returns a `DeviceInfo` object describing the first detected device:

```python
dev = wave_gpu.device()
print(dev.vendor)  # e.g. "Apple"
print(dev.name)    # e.g. "Apple M2 Max"
```

If no GPU is found the call raises a `RuntimeError`.

## Creating Arrays

`WaveArray` is the primary buffer type. It lives in device-accessible memory. Create one with the factory helpers:

```python
# From existing Python data
a = wave_gpu.array([1.0, 2.0, 3.0], dtype="f32")

# Pre-filled buffers
z = wave_gpu.zeros(1024, dtype="f32")
o = wave_gpu.ones(1024, dtype="f32")
```

Supported dtypes: `f16`, `f32`, `f64`, `i32`, `u32`.

`WaveArray` exposes a few useful properties and methods:

| Member | Description |
|---|---|
| `data` | Raw underlying data |
| `dtype` | Element type string |
| `len(arr)` | Number of elements (`__len__`) |
| `arr[i]` | Element access (`__getitem__`) |
| `arr.to_list()` | Copy contents back to a Python list |

## Writing Kernels

Decorate a plain Python function with `@wave_gpu.kernel` to mark it as a GPU kernel. Inside the kernel body you can use WAVE intrinsics to determine the current thread's position and synchronize:

```python
@wave_gpu.kernel
def vector_add(a, b, out, n):
    tid = wave_gpu.thread_id()
    if tid < n:
        out[tid] = a[tid] + b[tid]
```

### Available Intrinsics

| Intrinsic | Description |
|---|---|
| `thread_id()` | Global thread index |
| `workgroup_id()` | Workgroup (block) index |
| `lane_id()` | Lane within the current wave/warp |
| `wave_width()` | Number of lanes per wave |
| `barrier()` | Workgroup-level synchronization barrier |

Kernel parameters must be either `WaveArray` objects (for buffer access) or plain `int` / `float` values (for scalar uniforms).

## Launching Kernels

Call the decorated function directly. WAVE compiles the kernel on first invocation and caches the result. You can optionally specify the dispatch grid and workgroup size:

```python
n = 1024
a = wave_gpu.array([float(i) for i in range(n)], dtype="f32")
b = wave_gpu.ones(n, dtype="f32")
out = wave_gpu.zeros(n, dtype="f32")

# Simple launch - WAVE infers a 1-D grid from the first buffer
vector_add(a, b, out, n)

# Explicit grid and workgroup dimensions
vector_add(a, b, out, n, grid=(n // 256, 1, 1), workgroup=(256, 1, 1))
```

## Reading Results

After a kernel completes, read data back to the host with `to_list()`:

```python
result = out.to_list()
print(result[:8])  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

You can also index individual elements for quick spot-checks:

```python
assert out[0] == 1.0
```

## Error Handling

WAVE surfaces two main exception types:

- **`RuntimeError`** - raised when kernel compilation fails (e.g. unsupported intrinsic, invalid shader generation) or when no device is found.
- **`TypeError`** - raised when kernel arguments do not match the expected signature (wrong type or count).

Handle them with standard `try` / `except`:

```python
try:
    broken_kernel(a, b, out, n)
except RuntimeError as e:
    print(f"Compilation or device error: {e}")
except TypeError as e:
    print(f"Bad kernel arguments: {e}")
```

## Next Steps

See the full [Python API Reference](/reference/python-api) for detailed method signatures and advanced options.
