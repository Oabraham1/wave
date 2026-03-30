---
title: Pipeline Overview
description: The end-to-end WAVE compilation pipeline from source code to GPU execution.
---

WAVE transforms GPU kernels written in Python, Rust, C++, or TypeScript into vendor-native GPU code through a multi-stage compilation pipeline that preserves portability at every intermediate step.

## Full Pipeline

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

## Stage-by-Stage Breakdown

### 1. Source Code

Developers write GPU kernels using the WAVE SDK for their language of choice. Each SDK provides a `@wave.kernel` decorator or macro that marks a function for GPU compilation:

- **Python** (`wave-py`): Decorator-based API using `@wave.kernel` with NumPy-compatible types.
- **Rust** (`wave-rs`): Procedural macro `#[wave::kernel]` with native Rust types.
- **C++** (`wave-cpp`): Attribute-based annotation `[[wave::kernel]]` with standard C++ types.
- **TypeScript** (`wave-ts`): Decorator `@wave.kernel` with TypedArray-compatible types.

The SDK extracts the kernel function, serializes its AST or IR, and passes it to `wave-compiler`.

### 2. wave-compiler

The compiler is the core of the pipeline. It takes a language-specific kernel representation and produces a `.wbin` binary through six internal stages:

1. **Frontend** - Language-specific parsers produce a unified HIR.
2. **HIR (High-Level IR)** - Preserves source-level structure (loops, conditionals, variable names).
3. **MIR (Mid-Level IR)** - SSA-based IR suitable for optimization passes.
4. **Optimization** - DCE, CSE, SCCP, LICM, strength reduction, mem2reg, loop unrolling, CFG simplification. Controlled by optimization levels O0 through O3.
5. **LIR (Low-Level IR)** - Structured control flow with explicit register references.
6. **Register Allocation** - Chaitin-Briggs graph coloring with coalescing and spilling.
7. **Emission** - Encodes the LIR into the `.wbin` binary format.

See [Compiler Internals](/architecture/compiler/) for a detailed walkthrough of each stage.

### 3. .wbin Binary

The `.wbin` file is the portable artifact. It encodes WAVE's 11 hardware-invariant primitives into a compact binary format:

- **32-bit base instructions** for common operations (arithmetic, comparisons, register moves).
- **64-bit extended instructions** for operations requiring three or more operands, immediates, or memory addressing.
- **Metadata section** containing kernel name, register pressure, local memory requirements, and workgroup size hints.

A `.wbin` file contains no vendor-specific information. The same binary runs through any of the four backends or through the emulator.

### 4. Backend Translation

Each backend reads the `.wbin` and translates it to vendor-native code:

| Backend | Output | Target Hardware |
|---|---|---|
| `wave-metal` | Metal Shading Language (MSL) | Apple M1-M4, A-series GPUs |
| `wave-ptx` | PTX assembly | NVIDIA Turing+ GPUs (SM 75+) |
| `wave-hip` | HIP C++ | AMD RDNA / CDNA GPUs |
| `wave-sycl` | SYCL C++ | Intel Xe / Arc GPUs |

The backend output is then compiled by the vendor's own toolchain (`metallib`, `ptxas`, `hipcc`, `dpcpp`) into a GPU-executable binary.

See [Backends](/architecture/backends/) for a detailed comparison of translation strategies.

### 5. wave-emu (Emulator Path)

For CI pipelines and development machines without a GPU, `wave-emu` executes `.wbin` binaries directly on the CPU. It implements the full WAVE execution model including SIMT divergence, barrier synchronization, atomic operations, and wave-level collectives.

The emulator guarantees functional correctness but does not model vendor-specific performance characteristics. It is the reference implementation for the WAVE specification.

See [Emulator](/architecture/emulator/) for details on the execution model.

## How the Pieces Fit Together

The pipeline is designed around a single invariant: **the `.wbin` binary is the portability boundary**. Everything above it (SDKs, compiler frontend) is language-specific. Everything below it (backends, vendor toolchains) is hardware-specific. The `.wbin` format itself is neither.

This means:

- **Adding a new source language** requires only a new frontend parser in `wave-compiler`. The optimization pipeline, register allocator, backends, and emulator are reused unchanged.
- **Adding a new GPU vendor** requires only a new backend that translates `.wbin` to the vendor's native format. The compiler and all existing SDKs work without modification.
- **Testing without hardware** uses the same `.wbin` that would run on a real GPU. There is no separate "test mode" or "CPU fallback" compilation path.

The occupancy model is also portable. WAVE defines a universal occupancy equation:

```
O = floor(F / (R * W * w))
```

Where `F` is the register file size, `R` is the registers used per thread, `W` is the wave width, and `w` is the target number of waves per compute unit. Backends use this equation with vendor-specific values for `F` and `W` to determine launch configuration.

**Next:** [Compiler Internals](/architecture/compiler/) for a deep dive into the compilation stages.
