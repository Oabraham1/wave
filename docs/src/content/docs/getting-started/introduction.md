---
title: Introduction
description: What WAVE is, the problem it solves, and how it works.
---

WAVE (Wide Architecture Virtual Encoding) is a vendor-neutral GPU instruction set architecture that lets you write GPU kernels once and run them on Apple, NVIDIA, AMD, and Intel hardware without modification.

## The Problem: GPU Vendor Lock-In

GPU computing today is fragmented. NVIDIA's CUDA dominates with its mature ecosystem, but CUDA kernels only run on NVIDIA hardware. If you want to target Apple Silicon, you rewrite in Metal Shading Language. AMD requires ROCm/HIP. Intel requires oneAPI/SYCL. Each vendor defines its own ISA, memory model, synchronization primitives, and toolchain.

This means:

- **Duplicated engineering effort** - the same algorithm rewritten 2-4 times for different vendors.
- **Vendor lock-in** - choosing CUDA today locks out every non-NVIDIA GPU tomorrow.
- **Fragmented testing** - each port requires its own validation and performance tuning.
- **Limited portability** - researchers and startups cannot afford to maintain multiple GPU backends.

## The Insight: 11 Hardware-Invariant Primitives

WAVE was built on a systematic analysis of 5,000+ pages of vendor ISA documentation spanning 16 microarchitectures across all four major GPU vendors. That analysis revealed that despite surface-level differences in encoding, naming, and calling conventions, every GPU architecture provides the same 11 categories of fundamental operations:

| # | Primitive Category | Description | Examples |
|---|---|---|---|
| 1 | **Integer Arithmetic** | Addition, subtraction, multiplication, division, modulo on integer types | `iadd`, `imul`, `idiv` |
| 2 | **Floating-Point Arithmetic** | IEEE 754 arithmetic on 16/32/64-bit floats | `fadd`, `fmul`, `fdiv`, `fma` |
| 3 | **Bitwise Operations** | AND, OR, XOR, NOT, shifts, population count | `and`, `or`, `shl`, `shr` |
| 4 | **Comparison & Selection** | Compare two values, set predicates, conditional select | `cmp`, `sel` |
| 5 | **Local (Shared) Memory** | Read/write to workgroup-visible scratchpad memory | `lds_load`, `lds_store` |
| 6 | **Device (Global) Memory** | Read/write to device-wide memory with configurable scope | `load`, `store` |
| 7 | **Atomic Operations** | Atomic read-modify-write with scoped visibility | `atom_add`, `atom_cas` |
| 8 | **Wave/Warp Operations** | Subgroup-level shuffles, reductions, ballots | `wave_shuffle`, `wave_reduce` |
| 9 | **Control Flow** | Structured branches, loops, and function calls | `if/else/endif`, `loop/break/endloop` |
| 10 | **Synchronization** | Barriers and memory fences at multiple scopes | `barrier`, `fence` |
| 11 | **Type Conversion** | Widening, narrowing, float-int, and format conversions | `cvt_f32_i32`, `cvt_f16_f32` |

A twelfth capability - **special register access** - provides read-only access to hardware identity registers such as thread ID, workgroup ID, and workgroup dimensions via 16 special registers.

## How the Pipeline Works

WAVE defines a complete compilation pipeline from source to GPU execution:

```
Source kernel
    │
    ▼
wave-compiler        Parses WAVE assembly, validates, encodes
    │
    ▼
  .wbin              Portable binary (32-bit base / 64-bit extended instructions)
    │
    ▼
Backend              wave-metal | wave-ptx | wave-hip | wave-sycl
    │
    ▼
Vendor GPU code      Metal IR | PTX | HIP C++ | SYCL C++
    │
    ▼
  GPU                Dispatched to hardware via vendor runtime
```

### The Toolchain

The WAVE toolchain consists of four core tools:

- **wave-compiler** - Compiles WAVE assembly source into the `.wbin` binary format. Performs validation, register allocation checks, and encoding.
- **wave-asm** - Standalone assembler that converts WAVE assembly text into `.wbin` binaries.
- **wave-dis** - Disassembler that converts `.wbin` binaries back into human-readable WAVE assembly for inspection and debugging.
- **wave-emu** - Instruction-level emulator that executes `.wbin` binaries on the CPU without requiring a GPU. Supports all 11 primitive categories with cycle-accurate memory model semantics.

### The Backends

Four backends translate `.wbin` to vendor-native code:

| Backend | Target | Output |
|---|---|---|
| `wave-metal` | Apple GPUs (M1-M4, A-series) | Metal Shading Language / Metal IR |
| `wave-ptx` | NVIDIA GPUs (Turing, Ampere, Hopper) | PTX assembly |
| `wave-hip` | AMD GPUs (RDNA, CDNA) | HIP C++ |
| `wave-sycl` | Intel GPUs (Xe, Arc) | SYCL C++ |

### The Binary Format

WAVE uses a compact binary encoding:

- **Base instructions** are 32 bits wide and encode opcode, destination register, and up to two source operands.
- **Extended instructions** are 64 bits wide and support three or more operands, immediate values, and memory addressing modes.
- The register file provides **32 general-purpose registers** (`r0`-`r31`), **4 predicate registers** (`p0`-`p3`) for conditional execution, and **16 special registers** for hardware identity and configuration.

### The Memory Model

WAVE defines a scoped memory model with four visibility levels:

1. **Wave** - visible within a single wave/warp/simdgroup.
2. **Workgroup** - visible within a single workgroup/threadblock/threadgroup.
3. **Device** - visible across all workgroups on the device.
4. **System** - visible across the device and host CPU.

Memory operations and fences specify their scope explicitly, giving the programmer precise control over visibility and ordering.

## Research Foundation

The WAVE ISA is the product of peer-level research. The full specification, primitive derivation methodology, and cross-vendor verification results are published on Zenodo under DOI [10.5281/zenodo.19163452](https://doi.org/10.5281/zenodo.19163452). The paper is in preparation for submission to ASPLOS 2027.

Verification has been performed on three hardware platforms:

- Apple M4 Pro (Metal backend)
- NVIDIA T4 (PTX backend)
- AMD MI300X (HIP backend)

Each primitive was validated against the vendor's native ISA specification to ensure semantic equivalence.

**Next:** [Installation](/getting-started/installation/) - install the SDK for your language.
