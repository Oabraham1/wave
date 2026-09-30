---
title: Supported GPUs
description: WAVE hardware targets and emulator execution.
---

WAVE has backends for Apple, NVIDIA, AMD, and Intel GPUs. Hardware support depends on the operations a kernel uses and the installed vendor toolchain.

## Tested hardware

WAVE hardware verification includes these devices:

| Vendor | GPU | Architecture | Backend |
|---|---|---|---|
| Apple | M4 Pro | Apple GPU | `wave-metal` |
| NVIDIA | T4 | Turing, SM 7.5 | `wave-ptx` |
| AMD | MI300X | CDNA 3 | `wave-hip` |

The [hardware verification page](/research/hardware-verification/) describes the matched benchmark study on Apple M4 Pro and NVIDIA T4. Intel hardware verification is pending.

## Other devices

The backends target Metal, PTX, HIP, and SYCL rather than individual GPU models. Other devices that support these toolchains are candidate targets. Run correctness checks on the intended device before relying on its results. Wave width, numeric capabilities, and resource limits can differ between devices.

| Backend | Required vendor toolchain |
|---|---|
| `wave-metal` | Metal |
| `wave-ptx` | CUDA |
| `wave-hip` | ROCm |
| `wave-sycl` | Intel oneAPI |

## Emulator

The CPU emulator executes WAVE binaries without a GPU. Use it to test kernel behavior and compare outputs with hardware execution. It does not model GPU timing.

The [quick start](/getting-started/quickstart/) runs a kernel explicitly through the emulator. SDK execution paths and fallback behavior are described in the language guides.

## Device detection

The Python SDK reports the selected device:

```python
import wave_gpu

print(wave_gpu.device())
```

The Rust SDK exposes the same device information:

```rust
use wave_gpu::device;

fn main() -> Result<(), wave_gpu::RuntimeError> {
    let dev = device::detect()?;
    println!("{}: {:?}", dev.name, dev.vendor);
    Ok(())
}
```

See [Installation](/getting-started/installation/) for SDK setup and [Backends](/architecture/backends/) for translation details.
