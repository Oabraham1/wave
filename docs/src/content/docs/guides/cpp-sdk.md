---
title: C/C++ SDK
description: Getting started with the WAVE C/C++ SDK for portable GPU programming.
---

The WAVE C SDK exposes a plain-C API with opaque handles, making it easy to integrate into C or C++ projects and to wrap from other languages via FFI. This guide covers building the library, detecting a device, managing buffers, compiling kernels, and handling errors.

## Building from Source

WAVE ships as a static library built with CMake:

```bash
mkdir build && cd build
cmake ..
make
```

This produces `libwave.a`. Link it into your project and include the public header:

```c
#include <wave.h>
```

In your CMakeLists.txt:

```cmake
target_link_libraries(my_app PRIVATE wave)
```

## Device Detection

Call `wave_detect_device()` to obtain a device handle. The handle is an opaque pointer that you pass to subsequent API calls:

```c
wave_device_t *dev = wave_detect_device();
if (!dev) {
    fprintf(stderr, "No GPU found: %s\n", wave_last_error());
    return 1;
}

printf("vendor: %d\n", wave_device_vendor(dev));
printf("name:   %s\n", wave_device_name(dev));
```

`wave_device_vendor()` returns a `wave_vendor_t` enum value. `wave_device_name()` returns a null-terminated string owned by the device handle.

### wave_vendor_t Values

| Value | Meaning |
|---|---|
| `WAVE_VENDOR_APPLE` | Apple GPU (Metal) |
| `WAVE_VENDOR_NVIDIA` | NVIDIA GPU |
| `WAVE_VENDOR_AMD` | AMD GPU |
| `WAVE_VENDOR_INTEL` | Intel GPU |
| `WAVE_VENDOR_EMULATOR` | Software emulator |

## Creating Buffers

Buffers are created through typed factory functions. Each returns an opaque `wave_buffer_t*`:

```c
float host_data[] = {1.0f, 2.0f, 3.0f, 4.0f};

// From existing host data
wave_buffer_t *a = wave_create_buffer_f32(host_data, 4);

// Zero-initialized buffer
wave_buffer_t *out = wave_create_zeros_f32(1024);
```

A `NULL` return indicates an allocation failure; check `wave_last_error()` for details.

## Compiling Kernels

Pass kernel source code and a `wave_lang_t` constant to `wave_compile()`:

```c
const char *source =
    "import wave_gpu\n"
    "\n"
    "@wave_gpu.kernel\n"
    "def vector_add(a, b, out, n):\n"
    "    tid = wave_gpu.thread_id()\n"
    "    if tid < n:\n"
    "        out[tid] = a[tid] + b[tid]\n";

wave_kernel_t *kern = wave_compile(source, WAVE_LANG_PYTHON);
if (!kern) {
    fprintf(stderr, "Compile error: %s\n", wave_last_error());
    return 1;
}
```

### wave_lang_t Values

| Value | Language |
|---|---|
| `WAVE_LANG_PYTHON` | Python |
| `WAVE_LANG_RUST` | Rust |
| `WAVE_LANG_CPP` | C++ |
| `WAVE_LANG_TYPESCRIPT` | TypeScript |

## Launching Kernels

`wave_launch()` dispatches a compiled kernel on a device. You supply arrays of buffer pointers and scalar values, plus 3-element arrays for the grid and workgroup dimensions:

```c
uint32_t n = 1024;

wave_buffer_t *buffers[] = { a, b, out };
float scalars[] = { (float)n };
uint32_t grid[]      = { n / 256, 1, 1 };
uint32_t workgroup[] = { 256, 1, 1 };

int rc = wave_launch(kern, dev, buffers, 3, scalars, 1, grid, workgroup);
if (rc != 0) {
    fprintf(stderr, "Launch failed: %s\n", wave_last_error());
}
```

`wave_launch` returns `0` on success and `-1` on failure.

## Reading Results

Copy device data back to a host-allocated array with `wave_read_buffer_f32()`:

```c
float result[1024];
int rc = wave_read_buffer_f32(out, result, 1024);
if (rc != 0) {
    fprintf(stderr, "Read failed: %s\n", wave_last_error());
}
printf("result[0] = %f\n", result[0]);
```

The function returns `0` on success and `-1` on error. The caller is responsible for providing a buffer large enough to hold `count` elements.

## Cleanup

Every handle type has a corresponding free function. Always free handles when they are no longer needed to avoid resource leaks:

```c
wave_free_kernel(kern);
wave_free_buffer(a);
wave_free_buffer(b);
wave_free_buffer(out);
wave_free_device(dev);
```

## Error Handling

Fallible functions signal errors through their return value (`NULL` for pointer-returning functions, `-1` for int-returning functions). Call `wave_last_error()` immediately after a failure to retrieve a human-readable error message:

```c
wave_kernel_t *kern = wave_compile(bad_source, WAVE_LANG_PYTHON);
if (!kern) {
    const char *msg = wave_last_error();
    fprintf(stderr, "error: %s\n", msg);
}
```

`wave_last_error()` returns a pointer to a thread-local string. The pointer is valid until the next WAVE API call on the same thread.

## Next Steps

See the full [C/C++ API Reference](/reference/cpp-api) for the complete function listing and type definitions.
