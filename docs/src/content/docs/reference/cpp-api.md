---
title: C/C++ API Reference
description: Complete API reference for the WAVE C/C++ SDK.
---

The WAVE C API provides a low-level interface for GPU compute from C and C++ programs. The API is defined in `wave.h` and follows a handle-based design with explicit error checking.

## Installation

Link against `libwave` and include the header:

```c
#include <wave.h>
```

Compile with:

```bash
cc -o my_program my_program.c -lwave
```

---

## Types

### Opaque Handles

| Type | Description |
|------|-------------|
| `wave_device_t` | Handle to a detected GPU device. |
| `wave_buffer_t` | Handle to a device-resident buffer. |
| `wave_kernel_t` | Handle to a compiled kernel. |

All handles are pointer-sized opaque types. A `NULL` handle indicates an error - call `wave_last_error()` to retrieve the error message.

### Enumerations

#### `wave_vendor_t`

```c
typedef enum {
    WAVE_VENDOR_AMD     = 0,
    WAVE_VENDOR_NVIDIA  = 1,
    WAVE_VENDOR_INTEL   = 2,
    WAVE_VENDOR_UNKNOWN = 3,
} wave_vendor_t;
```

#### `wave_lang_t`

```c
typedef enum {
    WAVE_LANG_PYTHON     = 0,
    WAVE_LANG_RUST       = 1,
    WAVE_LANG_CPP        = 2,
    WAVE_LANG_TYPESCRIPT = 3,
} wave_lang_t;
```

#### `wave_dtype_t`

```c
typedef enum {
    WAVE_DTYPE_F16 = 0,
    WAVE_DTYPE_F32 = 1,
    WAVE_DTYPE_F64 = 2,
    WAVE_DTYPE_I32 = 3,
    WAVE_DTYPE_U32 = 4,
} wave_dtype_t;
```

---

## Error Handling

All functions that return a handle return `NULL` on failure. Functions that return `int` return `0` on success and `-1` on failure. In both cases, call `wave_last_error()` to get a human-readable error string.

### `wave_last_error`

```c
const char* wave_last_error(void);
```

Returns a pointer to a null-terminated string describing the most recent error. The string is valid until the next WAVE API call on the same thread. Returns `NULL` if no error has occurred.

**Example:**

```c
wave_device_t dev = wave_device_detect();
if (dev == NULL) {
    fprintf(stderr, "Error: %s\n", wave_last_error());
    return 1;
}
```

---

## Device Functions

### `wave_device_detect`

```c
wave_device_t wave_device_detect(void);
```

Detect the first available GPU device.

**Returns:** A `wave_device_t` handle, or `NULL` if no supported GPU is found.

### `wave_device_vendor`

```c
wave_vendor_t wave_device_vendor(wave_device_t device);
```

Get the vendor of a detected device.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `wave_device_t` | A valid device handle. |

**Returns:** A `wave_vendor_t` value.

### `wave_device_name`

```c
const char* wave_device_name(wave_device_t device);
```

Get the name string of a detected device. The returned pointer is valid for the lifetime of the device handle.

**Returns:** A null-terminated device name string.

### `wave_device_destroy`

```c
void wave_device_destroy(wave_device_t device);
```

Release a device handle and its associated resources.

---

## Buffer Functions

### `wave_buffer_create`

```c
wave_buffer_t wave_buffer_create(const void* data, size_t count, wave_dtype_t dtype);
```

Create a device buffer from host data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `const void*` | Pointer to host data. Element size is determined by `dtype`. |
| `count` | `size_t` | Number of elements. |
| `dtype` | `wave_dtype_t` | Element type. |

**Returns:** A `wave_buffer_t` handle, or `NULL` on failure.

```c
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
wave_buffer_t buf = wave_buffer_create(data, 4, WAVE_DTYPE_F32);
```

### `wave_buffer_zeros`

```c
wave_buffer_t wave_buffer_zeros(size_t count, wave_dtype_t dtype);
```

Create a zero-initialized device buffer.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `count` | `size_t` | Number of elements. |
| `dtype` | `wave_dtype_t` | Element type. |

**Returns:** A `wave_buffer_t` handle, or `NULL` on failure.

### `wave_buffer_read`

```c
int wave_buffer_read(wave_buffer_t buffer, void* dst, size_t count);
```

Copy buffer contents from device to host memory.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `buffer` | `wave_buffer_t` | Source device buffer. |
| `dst` | `void*` | Destination host pointer. Must have space for at least `count` elements. |
| `count` | `size_t` | Number of elements to read. |

**Returns:** `0` on success, `-1` on failure.

```c
float result[4];
if (wave_buffer_read(out, result, 4) != 0) {
    fprintf(stderr, "Read failed: %s\n", wave_last_error());
}
```

### `wave_buffer_count`

```c
size_t wave_buffer_count(wave_buffer_t buffer);
```

Returns the number of elements in the buffer.

### `wave_buffer_destroy`

```c
void wave_buffer_destroy(wave_buffer_t buffer);
```

Release a device buffer and free its device memory.

---

## Kernel Functions

### `wave_kernel_compile`

```c
wave_kernel_t wave_kernel_compile(const char* source, wave_lang_t lang);
```

Compile a kernel from source code.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `const char*` | Null-terminated source code string. |
| `lang` | `wave_lang_t` | Source language. |

**Returns:** A `wave_kernel_t` handle, or `NULL` on compilation failure.

```c
const char* src =
    "@wave_gpu.kernel\n"
    "def add(a: f32, b: f32, out: f32):\n"
    "    tid = thread_id()\n"
    "    out[tid] = a[tid] + b[tid]\n";

wave_kernel_t kernel = wave_kernel_compile(src, WAVE_LANG_PYTHON);
if (kernel == NULL) {
    fprintf(stderr, "Compile error: %s\n", wave_last_error());
    return 1;
}
```

### `wave_kernel_launch`

```c
int wave_kernel_launch(
    wave_kernel_t kernel,
    wave_device_t device,
    wave_buffer_t* buffers,
    size_t buffer_count,
    const uint32_t* scalars,
    size_t scalar_count,
    uint32_t grid[3],
    uint32_t workgroup[3]
);
```

Dispatch a compiled kernel on a device.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | `wave_kernel_t` | Compiled kernel handle. |
| `device` | `wave_device_t` | Target device. |
| `buffers` | `wave_buffer_t*` | Array of buffer handles, in kernel argument order. |
| `buffer_count` | `size_t` | Number of buffers. |
| `scalars` | `const uint32_t*` | Array of 32-bit scalar values, or `NULL`. |
| `scalar_count` | `size_t` | Number of scalars. |
| `grid` | `uint32_t[3]` | Grid dimensions `{x, y, z}`. |
| `workgroup` | `uint32_t[3]` | Workgroup dimensions `{x, y, z}`. |

**Returns:** `0` on success, `-1` on failure.

```c
wave_buffer_t bufs[] = {a, b, out};
uint32_t grid[] = {4, 1, 1};
uint32_t wg[]   = {4, 1, 1};

if (wave_kernel_launch(kernel, dev, bufs, 3, NULL, 0, grid, wg) != 0) {
    fprintf(stderr, "Launch failed: %s\n", wave_last_error());
}
```

### `wave_kernel_destroy`

```c
void wave_kernel_destroy(wave_kernel_t kernel);
```

Release a compiled kernel handle.

---

## Complete Example

```c
#include <stdio.h>
#include <wave.h>

int main(void) {
    wave_device_t dev = wave_device_detect();
    if (!dev) {
        fprintf(stderr, "No GPU: %s\n", wave_last_error());
        return 1;
    }

    printf("GPU: %s\n", wave_device_name(dev));

    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};

    wave_buffer_t a   = wave_buffer_create(a_data, 4, WAVE_DTYPE_F32);
    wave_buffer_t b   = wave_buffer_create(b_data, 4, WAVE_DTYPE_F32);
    wave_buffer_t out = wave_buffer_zeros(4, WAVE_DTYPE_F32);

    const char* src =
        "@wave_gpu.kernel\n"
        "def add(a: f32, b: f32, out: f32):\n"
        "    tid = thread_id()\n"
        "    out[tid] = a[tid] + b[tid]\n";

    wave_kernel_t k = wave_kernel_compile(src, WAVE_LANG_PYTHON);
    if (!k) {
        fprintf(stderr, "Compile: %s\n", wave_last_error());
        return 1;
    }

    wave_buffer_t bufs[] = {a, b, out};
    uint32_t grid[] = {4, 1, 1};
    uint32_t wg[]   = {4, 1, 1};

    wave_kernel_launch(k, dev, bufs, 3, NULL, 0, grid, wg);

    float result[4];
    wave_buffer_read(out, result, 4);
    for (int i = 0; i < 4; i++) printf("%.1f ", result[i]);
    printf("\n");

    wave_kernel_destroy(k);
    wave_buffer_destroy(a);
    wave_buffer_destroy(b);
    wave_buffer_destroy(out);
    wave_device_destroy(dev);

    return 0;
}
```
