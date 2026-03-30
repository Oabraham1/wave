# WAVE

**Wide Architecture Virtual Encoding: The Universal GPU ISA**

WAVE is a vendor-neutral instruction set architecture for general-purpose GPU computation. Write a kernel once, run it on Apple, NVIDIA, AMD, or Intel GPUs. The toolchain compiles high-level GPU kernels (Python, Rust, C++, TypeScript) to a portable binary format, then translates that binary to vendor-specific GPU code at launch time. WAVE has been verified on Apple M4 Pro, NVIDIA T4, and AMD MI300X hardware.

Spec: https://ojima.me/spec.html

## Quick Start

```python
import wave_gpu

@wave_gpu.kernel
def vector_add(a: wave_gpu.f32[:], b: wave_gpu.f32[:], out: wave_gpu.f32[:], n: wave_gpu.u32):
    gid = wave_gpu.thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]

a = wave_gpu.array([1.0, 2.0, 3.0, 4.0])
b = wave_gpu.array([5.0, 6.0, 7.0, 8.0])
out = wave_gpu.zeros(4)
vector_add(a, b, out, len(a))
print(out.to_list())  # [6.0, 8.0, 10.0, 12.0]
```

## Repository Structure

| Directory | What |
|-----------|------|
| `wave-decode` | Shared instruction decoder and WBIN binary format |
| `wave-asm` | Assembler (text → binary) |
| `wave-dis` | Disassembler (binary → text) |
| `wave-emu` | Reference emulator |
| `wave-compiler` | Multi-language compiler (Python/Rust/C++/TypeScript → WAVE) |
| `wave-metal` | Apple Metal backend |
| `wave-ptx` | NVIDIA PTX backend |
| `wave-hip` | AMD HIP backend |
| `wave-sycl` | Intel SYCL backend |
| `wave-runtime` | Shared runtime for all SDKs |
| `sdk/python` | Python SDK (`pip install wave-gpu`) |
| `sdk/rust` | Rust SDK |
| `sdk/cpp` | C/C++ SDK |
| `sdk/js` | TypeScript/JavaScript SDK |
| `tests/spec-verification` | Specification conformance tests |
| `tests/ci` | CI integration test scripts and kernels |

## Hardware Verification

| Vendor | GPU | Status |
|--------|-----|--------|
| Apple | M4 Pro | Verified |
| NVIDIA | T4 | Verified |
| AMD | MI300X | Verified |
| Intel | - | Pending |

## SDKs

```
pip install wave-gpu          # Python
cargo add wave-sdk            # Rust
cmake (see sdk/cpp/)          # C/C++
npm install wave-gpu          # TypeScript
```

## Build from Source

```bash
git clone https://github.com/Oabraham1/wave
cd wave
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime; do
  (cd $crate && cargo build --release)
done
```

## Research

- [Toward a Universal GPU Instruction Set Architecture: A Cross-Vendor Analysis of Hardware-Invariant Computational Primitives in Parallel Processors](https://doi.org/10.5281/zenodo.19163452) (Zenodo)

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](LICENSE) for terms.
