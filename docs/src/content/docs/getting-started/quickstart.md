---
title: "Quick start"
description: "Compile a vector-add kernel and execute it with the WAVE emulator."
---

This example compiles Python kernel source to a WAVE binary and runs it through the emulator. It works without a GPU. Build the toolchain and install the Python SDK using the [installation instructions](/getting-started/installation/).

## Vector addition

Save this as `vector_add.py`:

```python
import wave_gpu
from wave_gpu.runtime import compile_kernel, launch_emulator

source = """
@kernel
def vector_add(a: f32[:], b: f32[:], out: f32[:], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"""

a = wave_gpu.array([1.0, -2.5, 3.25, 4.0])
b = wave_gpu.array([5.0, 6.5, -7.25, 8.0])
out = wave_gpu.zeros(4)
binary = compile_kernel(source, "python")
launch_emulator(binary, [a, b, out], [4], (1, 1, 1), (4, 1, 1))
print(out.to_list())
```

Run it:

```sh
python3 vector_add.py
```

The result is `[6.0, 4.0, -4.0, 12.0]`. The launch uses one workgroup with four threads. The scalar argument `4` sets `n`; each thread adds one pair of elements.

## SDK guides

The [Python guide](/guides/python-sdk/) describes compilation and emulator execution. The [Rust guide](/guides/rust-sdk/) also covers launch through the vendor runtimes. See the [C++ guide](/guides/cpp-sdk/) and [TypeScript guide](/guides/typescript-sdk/) for their APIs.

For direct assembly, continue to [Writing kernels](/guides/writing-kernels/). For hardware requirements, see [Supported GPUs](/getting-started/supported-gpus/).
