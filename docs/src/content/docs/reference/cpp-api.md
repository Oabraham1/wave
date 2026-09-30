---
title: "C/C++ API reference"
description: "Types and functions in the WAVE C API."
---

The C API uses opaque handles declared in `wave/wave.h`. Buffers hold host data. The launch wrapper is incompatible with the current emulator CLI. Use the [Python quickstart](/getting-started/quickstart/) or [Rust SDK](/guides/rust-sdk/) for execution.

## Installation

Build and link the `wave` CMake target as shown in the [C/C++ guide](/guides/cpp-sdk/).

```c
#include <wave/wave.h>
```

## Types

### Opaque handles

| Type | Purpose |
|---|---|
| `wave_device_t*` | Detected device |
| `wave_buffer_t*` | Host buffer |
| `wave_kernel_t*` | Compiled kernel |

### Enumerations

#### `wave_vendor_t`

```c
typedef enum {
    WAVE_VENDOR_APPLE = 0,
    WAVE_VENDOR_NVIDIA = 1,
    WAVE_VENDOR_AMD = 2,
    WAVE_VENDOR_INTEL = 3,
    WAVE_VENDOR_EMULATOR = 4,
} wave_vendor_t;
```

#### `wave_lang_t`

```c
typedef enum {
    WAVE_LANG_PYTHON = 0,
    WAVE_LANG_RUST = 1,
    WAVE_LANG_CPP = 2,
    WAVE_LANG_TYPESCRIPT = 3,
} wave_lang_t;
```

#### `wave_dtype_t`

```c
typedef enum {
    WAVE_DTYPE_F32 = 0,
    WAVE_DTYPE_U32 = 1,
    WAVE_DTYPE_I32 = 2,
    WAVE_DTYPE_F16 = 3,
    WAVE_DTYPE_F64 = 4,
} wave_dtype_t;
```

The buffer functions below operate on `f32` data.

## Error handling

Compilation returns `NULL` on failure. Read and launch functions return `0` on success and `-1` on failure. Read the error immediately after a failed call.

### `wave_last_error`

```c
const char *wave_last_error(void);
```

Return the last error message, or `NULL` if none is recorded. The string belongs to the library and can change on subsequent API calls.

## Device functions

### `wave_detect_device`

```c
wave_device_t *wave_detect_device(void);
```

Detect a device, with an emulator fallback when no GPU is found. Free the handle with `wave_free_device`.

### `wave_device_vendor`

```c
wave_vendor_t wave_device_vendor(const wave_device_t *dev);
```

Return the vendor of a valid device handle.

### `wave_device_name`

```c
const char *wave_device_name(const wave_device_t *dev);
```

Return the device name. The string remains valid for the lifetime of the handle.

### `wave_free_device`

```c
void wave_free_device(wave_device_t *dev);
```

Release a device handle.

## Buffer functions

### `wave_create_buffer_f32`

```c
wave_buffer_t *wave_create_buffer_f32(const float *data, size_t count);
```

Copy `count` floats from host memory into a buffer.

### `wave_create_zeros_f32`

```c
wave_buffer_t *wave_create_zeros_f32(size_t count);
```

Create a buffer containing `count` zero-valued floats.

### `wave_read_buffer_f32`

```c
int wave_read_buffer_f32(const wave_buffer_t *buf, float *out, size_t count);
```

Copy `count` floats into `out`. The destination must have space for that many elements. Return `-1` if the type or count is invalid.

### `wave_free_buffer`

```c
void wave_free_buffer(wave_buffer_t *buf);
```

Release a buffer.

## Kernel functions

### `wave_compile`

```c
wave_kernel_t *wave_compile(const char *source, wave_lang_t lang);
```

Compile a null-terminated source string. Return `NULL` on failure.

### `wave_launch`

```c
int wave_launch(
    wave_kernel_t *kern,
    const wave_device_t *dev,
    wave_buffer_t **buffers,
    size_t n_buffers,
    const uint32_t *scalars,
    size_t n_scalars,
    const uint32_t grid[3],
    const uint32_t workgroup[3]
);
```

The wrapper attempts emulator execution with `--memory-file`, `--workgroup`, and `--reg`, which the current `wave-emu` CLI does not accept. Emulator launches return an error. Other device vendors are rejected directly. The signature above describes the callable interface, not a working execution path.

| Argument | Meaning |
|---|---|
| `buffers` | Buffer handles in kernel argument order |
| `n_buffers` | Number of buffers |
| `scalars` | Scalar arguments as `uint32_t` values, or `NULL` when empty |
| `n_scalars` | Number of scalar arguments |
| `grid` | Workgroup counts in x, y, and z |
| `workgroup` | Threads per workgroup in x, y, and z |

### `wave_free_kernel`

```c
void wave_free_kernel(wave_kernel_t *kern);
```

Release a compiled kernel.

## Complete example

This example creates and reads a buffer without dispatching a kernel:

```c
#include <stdio.h>
#include <wave/wave.h>

int main(void) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    wave_buffer_t *buffer = wave_create_buffer_f32(input, 4);
    if (!buffer) return 1;

    float output[4];
    int status = wave_read_buffer_f32(buffer, output, 4);
    if (status != 0) {
        fprintf(stderr, "%s\n", wave_last_error());
        wave_free_buffer(buffer);
        return 1;
    }

    for (size_t i = 0; i < 4; ++i) printf("%.1f\n", output[i]);
    wave_free_buffer(buffer);
    return 0;
}
```
