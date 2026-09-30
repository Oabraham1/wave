---
title: "C/C++ SDK"
description: "Build the C library and use its handle-based API."
---

The WAVE C API uses opaque handles for devices, host buffers, and compiled kernels.

## Building from source

From the repository root, build the static library with CMake:

```bash
cmake -S sdk/cpp -B sdk/cpp/build
cmake --build sdk/cpp/build
```

This produces `libwave.a`. Link it into your project and include the public header:

```c
#include <wave/wave.h>
```

In `CMakeLists.txt`:

```cmake
target_link_libraries(my_app PRIVATE wave)
```

## Device detection

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

### wave_vendor_t values

| Value | Meaning |
|---|---|
| `WAVE_VENDOR_APPLE` | Apple GPU; Metal |
| `WAVE_VENDOR_NVIDIA` | NVIDIA GPU |
| `WAVE_VENDOR_AMD` | AMD GPU |
| `WAVE_VENDOR_INTEL` | Intel GPU |
| `WAVE_VENDOR_EMULATOR` | Software emulator |

## Creating buffers

Buffers are created through typed factory functions. Each returns an opaque `wave_buffer_t*`:

```c
float host_data[] = {1.0f, 2.0f, 3.0f, 4.0f};

// From existing host data
wave_buffer_t *a = wave_create_buffer_f32(host_data, 4);

// Zero-initialized buffer
wave_buffer_t *out = wave_create_zeros_f32(1024);
```

A `NULL` return indicates an allocation failure; check `wave_last_error()` for details.

## Compiling kernels

Pass kernel source code and a `wave_lang_t` constant to `wave_compile()`:

```c
const char *source =
    "import wave_gpu\n"
    "\n"
    "@wave_gpu.kernel\n"
    "def vector_add(a: f32[:], b: f32[:], out: f32[:], n: u32):\n"
    "    tid = thread_id()\n"
    "    if tid < n:\n"
    "        out[tid] = a[tid] + b[tid]\n";

wave_kernel_t *kern = wave_compile(source, WAVE_LANG_PYTHON);
if (!kern) {
    fprintf(stderr, "Compile error: %s\n", wave_last_error());
    return 1;
}
```

### wave_lang_t values

| Value | Language |
|---|---|
| `WAVE_LANG_PYTHON` | Python |
| `WAVE_LANG_RUST` | Rust |
| `WAVE_LANG_CPP` | C++ |
| `WAVE_LANG_TYPESCRIPT` | TypeScript |

## Launching kernels

`wave_launch()` passes `--memory-file`, `--workgroup`, and `--reg` options that the current `wave-emu` CLI does not accept. Emulator launches fail, and GPU device vendors are rejected directly. Use the [Python quickstart](/getting-started/quickstart/) or [Rust SDK](/guides/rust-sdk/) for execution.

This call illustrates the arguments and error handling:

```c
uint32_t n = 1024;

wave_buffer_t *buffers[] = { a, b, out };
uint32_t scalars[] = { n };
uint32_t grid[]      = { n / 256, 1, 1 };
uint32_t workgroup[] = { 256, 1, 1 };

int rc = wave_launch(kern, dev, buffers, 3, scalars, 1, grid, workgroup);
if (rc != 0) {
    fprintf(stderr, "Launch failed: %s\n", wave_last_error());
}
```

`wave_launch` returns `0` on success and `-1` on failure.

## Reading results

Copy buffer contents into a caller-allocated array with `wave_read_buffer_f32()`:

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

Free each handle when it is no longer needed:

```c
wave_free_kernel(kern);
wave_free_buffer(a);
wave_free_buffer(b);
wave_free_buffer(out);
wave_free_device(dev);
```

## Error handling

Fallible functions return `NULL` for pointer results or `-1` for integer results. Call `wave_last_error()` immediately after a failure to retrieve a human-readable error message:

```c
wave_kernel_t *kern = wave_compile(bad_source, WAVE_LANG_PYTHON);
if (!kern) {
    const char *msg = wave_last_error();
    fprintf(stderr, "error: %s\n", msg);
}
```

`wave_last_error()` returns a pointer to a thread-local string. The pointer is valid until the next WAVE API call on the same thread.

## Next steps

See the [C/C++ API reference](/reference/cpp-api) for function signatures.
