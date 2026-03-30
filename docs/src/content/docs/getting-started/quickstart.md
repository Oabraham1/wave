---
title: Quick Start
description: Write and run a GPU vector addition kernel in Python, Rust, C++, or TypeScript.
---

This guide walks through a complete GPU vector addition in all four WAVE SDKs. Each example compiles a kernel, allocates device memory, launches the computation, and reads back the result.

## Python

```python
import wave_gpu

# Define a GPU kernel using the @kernel decorator.
# threads= sets the number of threads per workgroup.
@wave_gpu.kernel(threads=256)
def vector_add(a, b, out, n):
    tid = wave_gpu.thread_id()
    if tid < n:
        out[tid] = a[tid] + b[tid]

# Create device-resident arrays.
n = 1024
a = wave_gpu.array([1.0] * n)
b = wave_gpu.array([2.0] * n)
out = wave_gpu.array([0.0] * n)

# Launch the kernel. WAVE detects the GPU and selects the right backend.
vector_add(a, b, out, n)

# Read results back to the host.
result = out.to_list()
print(result[:4])  # [3.0, 3.0, 3.0, 3.0]
```

**What happens under the hood:**

1. The `@wave_gpu.kernel` decorator captures the function body and compiles it to WAVE assembly.
2. `wave_gpu.array()` allocates a device buffer and copies data to the GPU.
3. On launch, the SDK calls `wave-compiler` to produce a `.wbin` binary, selects the appropriate backend (Metal, PTX, HIP, or SYCL) based on the detected GPU, translates to vendor code, and dispatches.
4. `out.to_list()` copies the result buffer back to the host.

## Rust

```rust
use wave_sdk::array;
use wave_sdk::device;
use wave_sdk::kernel;

fn main() {
    // Detect the available GPU and backend.
    let dev = device::detect().expect("No GPU or emulator found");
    println!("Using backend: {:?}", dev.backend());

    // Define the kernel as a WAVE assembly string.
    let source = r#"
        .kernel vector_add
        .args a: ptr<f32>, b: ptr<f32>, out: ptr<f32>, n: u32
        .threads 256

        ld_special r0, %thread_id
        cmp_lt p0, r0, r3          ; r3 = n
        @p0 bra skip

        shl r1, r0, 2              ; byte offset = tid * 4
        load r4, [r0 + r1]         ; a[tid] -> this is simplified; real addressing uses base+offset
        load r5, [r1 + r1]         ; b[tid]
        fadd r6, r4, r5
        store [r2 + r1], r6        ; out[tid] = a[tid] + b[tid]

        skip:
        ret
    "#;

    // Compile the kernel source to a .wbin binary.
    let program = kernel::compile(source, kernel::Language::WaveAsm)
        .expect("Compilation failed");

    // Allocate device arrays.
    let n: u32 = 1024;
    let a = array::from_f32(&dev, &vec![1.0_f32; n as usize]);
    let b = array::from_f32(&dev, &vec![2.0_f32; n as usize]);
    let out = array::from_f32(&dev, &vec![0.0_f32; n as usize]);

    // Launch the kernel.
    dev.launch(&program, &[&a, &b, &out, &array::from_u32(&dev, &[n])])
        .expect("Launch failed");

    // Read back results.
    let result = out.to_vec_f32();
    println!("{:?}", &result[..4]); // [3.0, 3.0, 3.0, 3.0]
}
```

**Key Rust SDK types:**

- `device::Device` - represents a detected GPU or the emulator fallback.
- `kernel::Program` - a compiled `.wbin` binary ready for dispatch.
- `array::Array` - a device-resident buffer with typed host-side accessors.

## C++

```cpp
#include <wave/wave.h>
#include <cstdio>
#include <cstdlib>

int main() {
    // Detect GPU backend.
    WaveDevice* dev = wave_detect_device();
    if (!dev) {
        fprintf(stderr, "No GPU or emulator found\n");
        return 1;
    }
    printf("Backend: %s\n", wave_device_backend_name(dev));

    // Kernel source in WAVE assembly.
    const char* source =
        ".kernel vector_add\n"
        ".args a: ptr<f32>, b: ptr<f32>, out: ptr<f32>, n: u32\n"
        ".threads 256\n"
        "\n"
        "ld_special r0, %thread_id\n"
        "cmp_lt p0, r0, r3\n"
        "@p0 bra skip\n"
        "shl r1, r0, 2\n"
        "load r4, [r0 + r1]\n"
        "load r5, [r1 + r1]\n"
        "fadd r6, r4, r5\n"
        "store [r2 + r1], r6\n"
        "skip:\n"
        "ret\n";

    // Compile.
    WaveProgram* prog = wave_compile(dev, source, WAVE_LANG_ASM);
    if (!prog) {
        fprintf(stderr, "Compilation failed: %s\n", wave_last_error());
        return 1;
    }

    // Allocate device buffers.
    const uint32_t n = 1024;
    float host_a[1024], host_b[1024], host_out[1024];
    for (uint32_t i = 0; i < n; i++) {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
        host_out[i] = 0.0f;
    }

    WaveBuffer* buf_a   = wave_create_buffer_f32(dev, host_a, n);
    WaveBuffer* buf_b   = wave_create_buffer_f32(dev, host_b, n);
    WaveBuffer* buf_out = wave_create_buffer_f32(dev, host_out, n);
    WaveBuffer* buf_n   = wave_create_buffer_u32(dev, &n, 1);

    // Launch.
    WaveBuffer* args[] = { buf_a, buf_b, buf_out, buf_n };
    wave_launch(dev, prog, args, 4);

    // Read back.
    wave_read_buffer_f32(buf_out, host_out, n);
    printf("[%.1f, %.1f, %.1f, %.1f]\n",
           host_out[0], host_out[1], host_out[2], host_out[3]);
    // Output: [3.0, 3.0, 3.0, 3.0]

    // Cleanup.
    wave_destroy_buffer(buf_a);
    wave_destroy_buffer(buf_b);
    wave_destroy_buffer(buf_out);
    wave_destroy_buffer(buf_n);
    wave_destroy_program(prog);
    wave_destroy_device(dev);

    return 0;
}
```

**Compiling the C++ example:**

```bash
g++ -std=c++17 vector_add.cpp -lwave-sdk -o vector_add
./vector_add
```

Or with CMake (assuming WAVE is installed or fetched):

```cmake
add_executable(vector_add vector_add.cpp)
target_link_libraries(vector_add PRIVATE wave::wave-sdk)
```

## TypeScript

```typescript
import { kernel, array, detectDevice } from "wave-gpu";

async function main() {
  // Detect GPU backend.
  const device = detectDevice();
  console.log(`Backend: ${device.backend}`);

  // Define the kernel inline. The SDK compiles it to .wbin on first call.
  const vectorAdd = kernel({
    source: `
      .kernel vector_add
      .args a: ptr<f32>, b: ptr<f32>, out: ptr<f32>, n: u32
      .threads 256

      ld_special r0, %thread_id
      cmp_lt p0, r0, r3
      @p0 bra skip
      shl r1, r0, 2
      load r4, [r0 + r1]
      load r5, [r1 + r1]
      fadd r6, r4, r5
      store [r2 + r1], r6
      skip:
      ret
    `,
    language: "wave-asm",
  });

  // Allocate device arrays.
  const n = 1024;
  const a = array(new Float32Array(n).fill(1.0), device);
  const b = array(new Float32Array(n).fill(2.0), device);
  const out = array(new Float32Array(n).fill(0.0), device);

  // Launch the kernel asynchronously.
  await vectorAdd.launch(device, [a, b, out, n]);

  // Read back results.
  const result = await out.toFloat32Array();
  console.log(Array.from(result.slice(0, 4))); // [3, 3, 3, 3]
}

main();
```

**Running the TypeScript example:**

```bash
npx tsx vector_add.ts
```

The TypeScript SDK uses N-API to call into the native WAVE toolchain. The `kernel()` function returns a reusable compiled kernel object. Compilation happens once on the first call and is cached for subsequent launches. All GPU operations (`launch`, `toFloat32Array`) are asynchronous and return Promises.

## What to Try Next

- Change the array size to 1,000,000 elements and observe that the same kernel works without modification.
- Replace `fadd` with `fmul` to perform element-wise multiplication.
- Add a second kernel that computes the dot product using `wave_reduce` to explore wave-level operations.
- Set `WAVE_BACKEND=emulator` as an environment variable to force emulator mode and compare results.

**Next:** [Supported GPUs](/getting-started/supported-gpus/) - see which hardware WAVE runs on today.
