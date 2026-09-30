---
title: "Python SDK"
description: "Host arrays, kernel compilation, and emulator execution in Python."
---

The `wave_gpu` package compiles kernels and manages host arrays. Its direct launch path uses the emulator. For GPU dispatch, use the [Rust SDK](/guides/rust-sdk/).

## Installation

Install from PyPI:

```bash
pip install wave-gpu
```

Import the package in your code:

```python
import wave_gpu
```

## Device detection

`wave_gpu.device()` returns a `DeviceInfo` object for the detected device:

```python
dev = wave_gpu.device()
print(dev.vendor)  # e.g. "apple"
print(dev.name)    # e.g. "Apple GPU (Metal)"
```

If no GPU is found, detection returns the emulator.

## Creating arrays

`WaveArray` stores host data for kernel arguments. Create arrays with:

```python
# From existing Python data
a = wave_gpu.array([1.0, 2.0, 3.0], dtype="f32")

# Pre-filled buffers
z = wave_gpu.zeros(1024, dtype="f32")
o = wave_gpu.ones(1024, dtype="f32")
```

Supported dtypes: `f16`, `f32`, `f64`, `i32`, `u32`.

`WaveArray` provides:

| Member | Description |
|---|---|
| `data` | Raw underlying data |
| `dtype` | Element type string |
| `len(arr)` | Number of elements; `__len__` |
| `arr[i]` | Element access; `__getitem__` |
| `arr.to_list()` | Copy contents back to a Python list |

## Writing kernels

Decorate a plain Python function with `@wave_gpu.kernel` to mark it as a GPU kernel. Inside the kernel body you can use WAVE intrinsics to determine the current thread's position and synchronize:

```python
from wave_gpu import thread_id

@wave_gpu.kernel
def vector_add(a: wave_gpu.f32[:], b: wave_gpu.f32[:], out: wave_gpu.f32[:], n: wave_gpu.u32):
    tid = thread_id()
    if tid < n:
        out[tid] = a[tid] + b[tid]
```

### Available intrinsics

| Intrinsic | Description |
|---|---|
| `thread_id()` | Global thread index |
| `workgroup_id()` | Workgroup; block index |
| `lane_id()` | Lane within the current wave/warp |
| `wave_width()` | Number of lanes per wave |
| `barrier()` | Workgroup-level synchronization barrier |

Pass `WaveArray` objects as buffers and integers as scalar arguments. Float scalar arguments are converted to integers.

## Launching kernels

The decorated function compiles on first invocation and caches the binary. It selects the detected device and supports emulator dispatch. GPU devices raise `NotImplementedError` in this SDK. To select the emulator explicitly, use `compile_kernel` and `launch_emulator` as shown in [Quickstart](/getting-started/quickstart/).

On the emulator, call the function with optional grid and workgroup dimensions:

```python
n = 1024
a = wave_gpu.array([float(i) for i in range(n)], dtype="f32")
b = wave_gpu.ones(n, dtype="f32")
out = wave_gpu.zeros(n, dtype="f32")

# Infer a one-dimensional grid from the largest buffer
vector_add(a, b, out, n)

# Explicit grid and workgroup dimensions
vector_add(a, b, out, n, grid=(n // 256, 1, 1), workgroup=(256, 1, 1))
```

## Reading results

After execution, `to_list()` copies the array contents into a Python list:

```python
result = out.to_list()
print(result[:8])  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

Index individual elements with:

```python
assert out[0] == 1.0
```

## Error handling

Compilation and dispatch can raise:

- `RuntimeError`: compilation, translation, or execution failed.
- `TypeError`: an argument is neither an array nor a numeric scalar.

Handle them with standard `try` / `except`:

```python
try:
    broken_kernel(a, b, out, n)
except RuntimeError as e:
    print(f"Compilation or device error: {e}")
except TypeError as e:
    print(f"Bad kernel arguments: {e}")
```

## Next steps

See the [Python API reference](/reference/python-api) for method signatures.
