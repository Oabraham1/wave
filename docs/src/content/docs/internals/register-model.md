---
title: Register Model
description: Why WAVE uses untyped 32-bit registers and how this maps to real GPU register files.
---

WAVE registers are untyped 32-bit slots whose interpretation is determined entirely by the instruction that reads them, mirroring how GPU hardware actually works beneath the type systems of high-level shading languages.

## Untyped by Design

A single WAVE register `r5` can hold an integer, a floating-point value, or a memory address. There is no type tag, no runtime check, and no distinction at the register-file level:

```
iadd r0, r1, r2   ; treats r1, r2 as integers, writes integer result to r0
fadd r0, r1, r2   ; treats r1, r2 as IEEE 754 floats, writes float result to r0
load r0, [r1]     ; treats r1 as an address, loads 32 bits into r0
```

This is not an abstraction convenience --- it reflects hardware reality. NVIDIA, AMD, Intel, and Apple GPUs all use untyped vector register files. The ALU pipeline reinterprets bit patterns based on the opcode, not on register metadata. A typed register model would impose a fiction that every backend would then have to strip away.

## Why 32-Bit?

Thirty-two bits is the native register width across all four target architectures. While 64-bit values (doubles, long integers, pointers on some architectures) are supported, they occupy register pairs. This matches vendor behavior:

- **NVIDIA**: 32-bit registers; doubles use even/odd pairs.
- **AMD**: 32-bit VGPRs; 64-bit values occupy two consecutive VGPRs.
- **Intel**: 32-bit GRF elements; SIMD operations widen across elements.
- **Apple**: 32-bit registers in the SIMD-group register file.

Defining the base width as 32 bits means a WAVE register maps 1:1 to a physical register on every target, eliminating packing or splitting logic in the backend.

## Register Count

WAVE provides 32 registers per thread, addressed by the 5-bit register fields in the instruction encoding (see [Binary Encoding](/internals/binary-encoding/)). This count was chosen by analyzing register pressure across thousands of compute kernels:

- The median kernel uses 16--24 registers.
- Kernels exceeding 32 registers are rare and typically benefit from refactoring into smaller kernels.
- Keeping the count at 32 allows 5-bit encoding, which fits within the 32-bit instruction word without sacrificing bits for other fields.

The compiler's register allocator is responsible for mapping an unbounded number of virtual registers to 32 physical registers, spilling to local memory when necessary.

## Relationship to Occupancy

The register count directly determines occupancy --- the number of concurrent waves a compute unit can execute. Fewer registers per thread means more threads fit in the fixed-size register file. This relationship is formalized in the [Occupancy Equation](/internals/occupancy-equation/).

## Vendor Register File Comparison

| Property | NVIDIA (Ampere) | AMD (RDNA 3) | Intel (Xe-HPG) | Apple (M1) |
|----------|----------------|--------------|----------------|------------|
| Register width | 32-bit | 32-bit | 32-bit | 32-bit |
| Max registers/thread | 255 | 256 | 128 | 128 |
| File size per CU/SM/EU | 256 KB | 512 KB | 128 KB | 208 KB |
| Typing | Untyped | Untyped | Untyped | Untyped |

All four vendors confirm the same fundamental design: large, untyped, 32-bit register files. WAVE's register model is a direct reflection of this hardware consensus.
