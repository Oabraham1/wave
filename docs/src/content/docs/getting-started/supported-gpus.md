---
title: Supported GPUs
description: Hardware compatibility table for WAVE backends, plus emulator fallback for development.
---

WAVE targets four GPU vendor families through dedicated compiler backends. This page lists the GPUs that have been verified, those expected to work based on architecture compatibility, and the emulator fallback for development without a GPU.

## Verified Hardware

These GPUs have passed the full WAVE spec-verification test suite, confirming correct behavior for all 11 primitive categories.

| Vendor | GPU | Architecture | Backend | Status |
|--------|-----|-------------|---------|--------|
| Apple | M4 Pro | Apple GPU (Metal 3) | `wave-metal` | Verified |
| NVIDIA | T4 | Turing (SM 7.5) | `wave-ptx` | Verified |
| AMD | MI300X | CDNA 3 | `wave-hip` | Verified |

## Expected Compatible Hardware

These GPUs share the same architecture family as verified hardware and are expected to work. They have not yet been tested with the full spec-verification suite.

| Vendor | GPU | Architecture | Backend | Status |
|--------|-----|-------------|---------|--------|
| Apple | M1 | Apple GPU (Metal 3) | `wave-metal` | Expected |
| Apple | M1 Pro / Max / Ultra | Apple GPU (Metal 3) | `wave-metal` | Expected |
| Apple | M2 | Apple GPU (Metal 3) | `wave-metal` | Expected |
| Apple | M2 Pro / Max / Ultra | Apple GPU (Metal 3) | `wave-metal` | Expected |
| Apple | M3 | Apple GPU (Metal 3) | `wave-metal` | Expected |
| Apple | M3 Pro / Max / Ultra | Apple GPU (Metal 3) | `wave-metal` | Expected |
| Apple | M4 | Apple GPU (Metal 3) | `wave-metal` | Expected |
| Apple | M4 Max / Ultra | Apple GPU (Metal 3) | `wave-metal` | Expected |
| NVIDIA | RTX 2060-2080 Ti | Turing (SM 7.5) | `wave-ptx` | Expected |
| NVIDIA | RTX 3060-3090 Ti | Ampere (SM 8.6) | `wave-ptx` | Expected |
| NVIDIA | A100 | Ampere (SM 8.0) | `wave-ptx` | Expected |
| NVIDIA | RTX 4060-4090 | Ada Lovelace (SM 8.9) | `wave-ptx` | Expected |
| NVIDIA | H100 | Hopper (SM 9.0) | `wave-ptx` | Expected |
| AMD | RX 7900 XTX | RDNA 3 | `wave-hip` | Expected |
| AMD | RX 7600 | RDNA 3 | `wave-hip` | Expected |
| AMD | MI250X | CDNA 2 | `wave-hip` | Expected |
| Intel | Arc A770 | Xe HPG (Alchemist) | `wave-sycl` | Pending |
| Intel | Data Center GPU Max (Ponte Vecchio) | Xe HPC | `wave-sycl` | Pending |

**Status definitions:**

- **Verified** - all 11 primitive categories pass the spec-verification test suite on this hardware.
- **Expected** - same architecture family as a verified GPU; not yet tested.
- **Pending** - backend implementation is complete but hardware access for verification is not yet available.

## Emulator Fallback

When no supported GPU is detected, WAVE automatically falls back to `wave-emu`, a CPU-based instruction-level emulator. The emulator executes the same `.wbin` binary that would run on a GPU, so kernels behave identically regardless of whether they are running on hardware or in emulation.

The emulator is useful for:

- **Development** - write and debug kernels on a laptop without a discrete GPU.
- **CI/CD** - run the full test suite in cloud environments that lack GPU instances.
- **Correctness testing** - compare emulator output against hardware output to detect backend translation bugs.

To force emulator mode even when a GPU is available:

```bash
# Environment variable (works with all SDKs)
WAVE_BACKEND=emulator python my_kernel.py
```

```python
# Python SDK - explicit backend selection
import wave_gpu
device = wave_gpu.device(backend="emulator")
```

```rust
// Rust SDK - explicit backend selection
let dev = wave_sdk::device::with_backend(wave_sdk::Backend::Emulator);
```

The emulator runs single-threaded by default. Set `WAVE_EMU_THREADS` to enable multi-threaded emulation for larger workloads:

```bash
WAVE_EMU_THREADS=8 WAVE_BACKEND=emulator python my_kernel.py
```

## Checking Your Hardware

Every SDK provides a detection function that reports the selected backend:

```bash
# CLI
wave-emu --detect

# Python
python -c "import wave_gpu; print(wave_gpu.detect_backend())"

# Rust (in a binary)
wave_sdk::device::detect().map(|d| println!("{:?}", d.backend()));
```

Possible output values: `metal`, `ptx`, `hip`, `sycl`, `emulator`.

**Next:** [Introduction to the ISA](/architecture/overview/) - learn how WAVE instructions are encoded and executed.
