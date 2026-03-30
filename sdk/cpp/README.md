# WAVE C/C++ SDK

Write GPU kernels in C/C++, run on any GPU.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage

```c
#include <wave/wave.h>

int main() {
    wave_device_t dev = wave_detect_device();
    float a[] = {1.0, 2.0, 3.0, 4.0};
    float b[] = {5.0, 6.0, 7.0, 8.0};
    float out[4] = {0};

    wave_buffer_t buf_a = wave_create_buffer_f32(a, 4);
    wave_buffer_t buf_b = wave_create_buffer_f32(b, 4);
    wave_buffer_t buf_out = wave_create_buffer_f32(out, 4);

    const char* src = "__kernel void vector_add(float* a, float* b, float* out, uint32_t n) {\n"
                      "    uint32_t gid = thread_id();\n"
                      "    if (gid < n) { out[gid] = a[gid] + b[gid]; }\n"
                      "}\n";

    wave_kernel_t kern = wave_compile(src, WAVE_LANG_CPP);
    wave_buffer_t bufs[] = {buf_a, buf_b, buf_out};
    uint32_t scalars[] = {4};
    wave_launch(kern, &dev, bufs, 3, scalars, 1, (uint32_t[3]){1,1,1}, (uint32_t[3]){256,1,1});
    wave_read_buffer_f32(buf_out, out, 4);

    wave_free_buffer(buf_a);
    wave_free_buffer(buf_b);
    wave_free_buffer(buf_out);
    wave_free_kernel(kern);
    return 0;
}
```

## License

Apache 2.0 - see [LICENSE](../../LICENSE)
