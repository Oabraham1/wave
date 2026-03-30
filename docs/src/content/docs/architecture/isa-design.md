---
title: ISA Design
description: The research methodology and design principles behind WAVE's 11 hardware-invariant primitives.
---

The WAVE ISA defines 11 hardware-invariant primitive categories derived from a systematic analysis of every major GPU vendor's instruction set architecture, covering 5,000+ pages of documentation across 16 microarchitectures.

## Design Principle

WAVE follows a single governing principle: **define what hardware must do, not how it does it.**

Every WAVE instruction describes an operation in terms of its inputs, outputs, and observable side effects. It never prescribes a specific execution mechanism. For example, `barrier` means "all threads in the workgroup must reach this point before any may proceed" - it says nothing about how the hardware implements the wait (spinning, yielding, scoreboard, etc.).

This principle produces a **thin abstraction**: thick enough to be portable across vendors, thin enough to compile to efficient native code. Thick abstractions (like high-level compute languages) sacrifice performance. No abstraction at all (like raw PTX or GCN) sacrifices portability. WAVE sits at the ISA level - the lowest point where all vendors converge.

## Analysis Methodology

The WAVE ISA was derived by exhaustive cross-vendor analysis rather than by design committee or by generalizing from a single vendor.

### Source Documentation

The analysis covered official ISA documentation from all four major GPU vendors:

| Vendor | Documentation | Microarchitectures |
|---|---|---|
| NVIDIA | PTX ISA v1.0 through v9.2 | Tesla, Fermi, Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper |
| AMD | RDNA 1-4, CDNA 1-4 ISA references | GCN, RDNA, CDNA families |
| Intel | Gen11, Xe-LP, Xe-HPG, Xe-HPC programmer's references | Gen11 through Ponte Vecchio |
| Apple | Metal Shading Language specification, Metal Feature Set tables | A-series (G13), M1-M4 |

Total volume: approximately 5,000 pages across 16 distinct microarchitectures.

### Extraction Process

For each vendor's ISA, the analysis extracted:

1. **Operation categories** - what fundamental operations does the ISA provide?
2. **Operand models** - how are registers, memory, and immediates addressed?
3. **Memory hierarchy** - what memory spaces exist and what are their visibility rules?
4. **Synchronization primitives** - what mechanisms exist for ordering and coordinating threads?
5. **Execution model** - how are threads grouped, scheduled, and divergence handled?

### Convergence Analysis

The extracted features were aligned across vendors to identify convergence points - operations that every vendor provides, even if the encoding, naming, or specific semantics differ. An operation qualified as a hardware-invariant primitive only if:

- Every analyzed vendor provides it (universality requirement).
- It operates at the ISA level, not as a software library function (hardware requirement).
- It cannot be decomposed into a combination of other primitives without performance loss (irreducibility requirement).

## The 11 Primitives

The analysis produced 11 categories of hardware-invariant operations:

### 1. Lockstep Execution (Waves)

Every GPU groups threads into fixed-width SIMD lanes that share a program counter and execute in lockstep. NVIDIA calls them warps, AMD calls them wavefronts, Intel calls them subgroups, Apple calls them SIMD-groups. WAVE calls them **waves**.

The wave is the fundamental unit of scheduling. All GPU architectures provide it because the SIMT execution model depends on it - without lockstep grouping, GPU parallelism would require explicit thread management at every instruction.

### 2. SIMT Divergence Handling

When threads within a wave take different branch paths, the hardware must serialize execution of each path while tracking which threads are active. Every vendor implements a mechanism for this (masked execution with a reconvergence stack or equivalent).

WAVE models divergence with an **active mask** and **structured control flow** (`if/else/endif`, `loop/break/endloop`). This structured model is the subset that all vendors support - NVIDIA's recently-introduced independent thread scheduling extends beyond it, but structured divergence remains the universal baseline.

### 3. Register File

Every GPU provides a per-thread register file for arithmetic operands. The size, width, and organization differ:

| Vendor | Width | File size |
|---|---|---|
| NVIDIA | 32-bit | 256 per thread (configurable) |
| AMD | 32-bit (VGPR) + 32-bit (SGPR) | Varies by arch |
| Intel | 256-bit (GRF) | 128 registers |
| Apple | 32-bit | Not publicly documented |

WAVE defines a **32-register, 32-bit file** - the intersection that maps efficiently to all vendors. Wider operations (f64) use register pairs. Backends map WAVE registers to vendor-native registers during translation.

### 4. Local (Shared) Memory

A fast, workgroup-visible scratchpad memory. Every GPU architecture provides it:

- NVIDIA: shared memory (configurable vs. L1)
- AMD: LDS (Local Data Share)
- Intel: SLM (Shared Local Memory)
- Apple: threadgroup memory

WAVE models local memory as a byte-addressable array with workgroup lifetime and workgroup visibility.

### 5. Device (Global) Memory

Byte-addressable memory visible to all threads on the device. All vendors provide global load/store instructions with various width options. WAVE models device memory as a flat address space with 64-bit addressing.

### 6. Scoped Synchronization

Barriers and fences with explicit scope. Every vendor provides workgroup-level barriers. Fences with wider scope (device, system) are also universally available, though the mechanisms differ:

- NVIDIA: `bar.sync`, `membar.{cta,gl,sys}`
- AMD: `s_barrier`, `s_waitcnt`
- Intel: `barrier`, `fence`
- Apple: `threadgroup_barrier`

WAVE defines four scopes (wave, workgroup, device, system) and two operations (`barrier` for execution synchronization, `fence` for memory ordering).

### 7. Structured Control Flow

Every GPU ISA provides conditional branch and loop constructs. While the encoding differs (NVIDIA uses explicit branch targets, AMD uses scalar branch instructions, Intel uses structured EU control flow), all support the structured subset of `if/else/endif` and `loop/break/endloop`.

WAVE uses structured control flow exclusively. Arbitrary `goto`-style branches are not part of the ISA because not all vendors support them at the hardware level - some (notably Intel and Apple) require structured control flow.

### 8. Atomic Operations

Read-modify-write operations with guaranteed atomicity. Every vendor provides atomics on both local and device memory. The common set includes: add, subtract (or negate-add), AND, OR, XOR, min, max, exchange, and compare-and-swap.

WAVE defines all of these with explicit scope annotations. Backends translate to vendor-specific instructions, working around gaps (e.g., PTX lacks `atom.sub`, so the backend emits `neg` + `atom.add`).

### 9. Wave-Level Operations

Operations that communicate values across lanes within a wave. Every vendor provides:

- **Shuffle** - read a value from another lane.
- **Reduction** - combine values across all lanes (sum, min, max).
- **Ballot** - produce a bitmask of per-lane predicates.

These operations exist in hardware because they enable critical GPU algorithms (parallel prefix sum, warp-level reduction) without going through shared memory.

### 10. Type Conversion

Widening, narrowing, and format conversions between integer and floating-point types. Every vendor provides dedicated conversion instructions (e.g., `cvt.f32.s32` in PTX, `v_cvt_f32_i32` in AMD GCN).

WAVE defines explicit conversion instructions between all supported type pairs: `i32`, `u32`, `f32`, `f16`, `f64`, and `bool`.

### 11. Special Registers

Read-only hardware identity registers that provide:

- Thread ID within the workgroup (x, y, z)
- Workgroup ID within the grid (x, y, z)
- Workgroup dimensions (x, y, z)
- Grid dimensions
- Wave width
- Lane ID within the wave

Every vendor provides these values, though the access mechanism varies (NVIDIA uses special registers like `%tid.x`, AMD uses scalar registers loaded at dispatch, Intel uses EU thread payload registers, Apple uses `thread_position_in_threadgroup`).

WAVE defines 16 special registers (`sr_tid_x` through `sr_wave_width`) providing uniform access.

## What Was Excluded

The following capabilities were present in one or more vendor ISAs but excluded from WAVE because they failed the universality, hardware, or irreducibility requirements:

### Tensor/Matrix Operations

NVIDIA's Tensor Cores, AMD's Matrix Cores, and Intel's XMX engines provide hardware-accelerated matrix multiply-accumulate. These were excluded because:

- Apple GPUs do not provide equivalent hardware-level matrix operations.
- The matrix tile sizes and data types differ significantly across vendors (NVIDIA: 16x16x16, AMD: 32x32, Intel: 8x16).
- They can be decomposed into standard arithmetic on the WAVE primitives (at a performance cost, which is acceptable for a portability-first ISA).

### Texture Sampling

All GPUs provide texture sampling hardware, but the interface is deeply vendor-specific (descriptor formats, filtering modes, coordinate systems). Texture operations are also higher-level than ISA primitives - they are built on top of memory loads with hardware interpolation.

### Ray Tracing

NVIDIA RT Cores, AMD Ray Accelerators, and Intel Ray Tracing Units provide hardware ray-box and ray-triangle intersection. These were excluded because support is not universal across all microarchitectures within each vendor (let alone across vendors) and the interfaces are not yet stable.

### Asynchronous Copy

NVIDIA's `cp.async` and AMD's equivalent provide asynchronous memory copy from global to shared memory. These were excluded because Apple and older Intel architectures do not provide hardware-level async copy, and the operation can be decomposed into load + barrier + store sequences.

### Variable Wave Width

Some AMD architectures support runtime wave width selection (wave32 vs. wave64). WAVE treats wave width as a configuration parameter rather than an ISA-level construct - the emulator and backends handle it, but the ISA instructions themselves are wave-width-agnostic.

## Universal Occupancy Equation

The analysis revealed that despite architectural differences, every vendor's occupancy model reduces to the same equation:

```
O = floor(F / (R * W * w))
```

Where:

- **F** = Total register file size per compute unit (vendor-specific constant)
- **R** = Registers used per thread (determined by the compiler, reported in `.wbin` metadata)
- **W** = Wave width (vendor-specific: 32 for NVIDIA/Apple, 32 or 64 for AMD, variable for Intel)
- **w** = Target waves per compute unit (tuning parameter)

This equation governs how many waves can execute concurrently on a single compute unit. Backends use it with vendor-specific values for `F` and `W` to calculate optimal launch configurations.

The universality of this equation is a direct consequence of the register file primitive - since every vendor provides a per-thread register file that is partitioned across concurrent waves, the occupancy tradeoff between register usage and parallelism is structurally identical everywhere.
