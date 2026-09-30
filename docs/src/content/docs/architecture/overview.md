---
title: "Pipeline overview"
description: "The end-to-end WAVE compilation pipeline from source code to GPU execution."
---

WAVE compiles Python, Rust, C++, and TypeScript kernels to `.wbin`. A backend translates that binary for the target GPU.

## Full pipeline

```
Source Code                    Language-specific SDK
(Python / Rust / C++ / TS)     wave-py / wave-rs / wave-cpp / wave-ts
        │
        ▼
 ┌──────────────┐
 │ wave-compiler │              Frontend → HIR → MIR → Optimize → LIR → RegAlloc → Emit
 └──────┬───────┘
        │
        ▼
      .wbin                     Portable WAVE binary (hardware-invariant)
        │
        ├───────────────────────────────────────────┐
        │                                           │
        ▼                                           ▼
 ┌─────────────┐                             ┌──────────┐
 │   Backend    │                             │ wave-emu │
 │ (one of 4)  │                             └────┬─────┘
 └──────┬──────┘                                  │
        │                                         ▼
        ▼                                   CPU execution
  Vendor code                               (CI / testing)
  (MSL / PTX / HIP C++ / SYCL C++)
        │
        ▼
  Vendor toolchain
  (metallib / ptxas / hipcc / dpcpp)
        │
        ▼
    GPU execution
```

## Stage-by-Stage breakdown

### 1. source code

The language SDKs pass kernel source to `wave-compiler`. Each frontend accepts a subset of its source language. See the SDK guides for supported syntax and launch APIs.

### 2. wave-compiler

The compiler produces `.wbin` through these stages:

1. Frontends parse each supported source language into HIR.
2. HIR preserves loops, conditionals, and variable names.
3. MIR uses SSA form for optimization.
4. Optimization applies the passes enabled at levels O0 through O3.
5. LIR represents structured control flow with explicit registers.
6. Register allocation colors an interference graph while preserving parameter indices. Kernels that exceed the limit are rejected.
7. Emission encodes LIR into `.wbin`.

See [Compiler internals](/architecture/compiler/) for a guide to each stage.

### 3. .wbin binary

The `.wbin` container stores instruction words and kernel metadata:

- 32-bit base instructions contain an opcode, register fields, modifiers, and predication.
- 64-bit extended instructions add operand registers, immediates, or scope information.
- Metadata declares kernel names, register counts, local memory, and workgroup dimensions.

The same format feeds the four backends and the emulator. Execution requires the selected backend to support the kernel's operations, capabilities, and resource needs.

### 4. backend translation

Each backend translates `.wbin` into input for a vendor compiler:

| Backend | Output | Target hardware |
|---|---|---|
| `wave-metal` | Metal Shading Language; MSL | Apple M1-M4, A-series GPUs |
| `wave-ptx` | PTX assembly | NVIDIA Turing+ GPUs; SM 75+ |
| `wave-hip` | HIP C++ | AMD RDNA / CDNA GPUs |
| `wave-sycl` | SYCL C++ | Intel Xe / Arc GPUs |

Metal, CUDA, ROCm, or oneAPI tools compile the generated program for GPU execution.

See [Backends](/architecture/backends/) for a detailed comparison of translation strategies.

### 5. wave-emu: emulator path

`wave-emu` executes `.wbin` on the CPU for development and CI. It models lane state, divergence, barriers, atomics, and wave collectives.

The emulator is the reference implementation for functional tests. It does not model GPU timing.

See [Emulator](/architecture/emulator/) for details on the execution model.

## How the pieces fit together

The `.wbin` format connects language frontends to hardware backends.

- Add a frontend to support another source language.
- Add a backend to target another GPU vendor.
- Run the same binary in the emulator for CPU-based tests.

Register capacity gives an upper bound on resident waves:

```
O = floor(F / (R * W * w))
```

`F` is register storage in bytes, `R` is machine registers per lane, `W` is lanes per wave, and `w` is bytes per register. Hardware limits and allocation granularity can reduce occupancy further. WAVE metadata records virtual register demand; the vendor compiler determines machine register use.

Next: [Compiler internals](/architecture/compiler/) for details on the compilation stages.
