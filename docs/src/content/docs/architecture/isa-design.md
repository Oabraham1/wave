---
title: "ISA design"
description: "The operation categories, parameters, and capabilities in the WAVE ISA."
---

WAVE groups GPU compute operations into 11 primitive categories. Its design compares the execution interfaces of NVIDIA, AMD, Intel, and Apple.

## Design principle

An instruction defines its inputs, outputs, participating lanes, and observable effects. The backend chooses a target mechanism that preserves them. A workgroup barrier, for example, requires participating threads to reach a rendezvous; it does not prescribe how hardware tracks arrival.

This separation keeps memory scope, lane participation, and numerical behavior explicit while leaving machine encoding and scheduling to the target.

## Analysis methodology

The cross-vendor analysis compares operation semantics and the parameters needed to express them.

### Source documentation

| Vendor | Documentation | Microarchitectures |
|---|---|---|
| NVIDIA | PTX ISA v1.0 through v9.2 | Tesla, Fermi, Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper |
| AMD | RDNA 1-4, CDNA 1-4 ISA references | GCN, RDNA, CDNA families |
| Intel | Gen11, Xe-LP, Xe-HPG, Xe-HPC programmer's references | Gen11 through Ponte Vecchio |
| Apple | Metal Shading Language specification and feature tables | A-series, M1-M4 |

The research describes a survey of approximately 5,000 pages across 16 microarchitectures. These sources operate at different abstraction levels: a language interface does not document a vendor's machine encoding.

### Extraction process

The analysis compares five properties:

1. Operations and their observable effects.
2. Register, memory, and immediate operands.
3. Memory spaces and visibility.
4. Ordering and rendezvous requirements.
5. Thread grouping and conditional participation.

### Convergence analysis

A common operation needs a definition that each target can preserve. Matching names do not establish matching semantics. Widths and capacities become parameters; optional operations and unsupported scopes require capability checks.

## The 11 primitives

### 1. Lockstep execution: waves

WAVE groups lanes into waves with width `W`. NVIDIA calls these groups warps, AMD calls them wavefronts, Intel uses subgroups, and Metal uses SIMD-groups. The target may implement instruction scheduling differently while preserving the program's lane behavior.

### 2. SIMT divergence handling

An active mask records which lanes execute. Structured branches and loops change that mask. Backends preserve lane participation through the target's control-flow mechanism.

### 3. Register file

General-purpose registers contain untyped 32-bit values. Eight-bit fields address `r0` through `r255`; kernel metadata declares the required count. Wider values use consecutive registers. The vendor compiler assigns machine registers after translation.

### 4. Local: shared memory

Local memory is a byte-addressed region owned by one workgroup. It maps to NVIDIA shared memory, AMD LDS, Intel SLM, or Metal threadgroup memory. Threads synchronize accesses when they exchange values through it.

### 5. Device: global memory

Device memory is shared across workgroups. The implementation supplies a device-memory base pointer, and instructions use byte offsets to access it. Access width and address validity are part of the instruction contract.

### 6. Scoped synchronization

WAVE defines wave, workgroup, device, and system scopes. Barriers coordinate arrival; fences order memory accesses. A backend must implement the requested scope or reject it. Metal, for example, rejects system-scope fences.

### 7. Structured control flow

WAVE expresses branches with `if/else/endif` and loops with `loop/break/endloop`. The structure defines mask changes and reconvergence. Backends translate it into target control flow.

### 8. Atomic operations

Atomic operations read and update a memory location indivisibly. The set includes add, subtract, AND, OR, XOR, minimum, maximum, exchange, and compare-and-swap. A lowering can require several instructions: PTX subtraction uses negation followed by atomic addition, and Metal strong compare-and-swap retries a weak exchange after spurious failure.

### 9. Wave-level operations

Shuffle exchanges register values, reductions combine lane values, and ballots collect predicates. These operations make communication within a wave explicit. Their participation and source-lane requirements must be preserved.

### 10. Type conversion

Conversion instructions specify both source and destination formats. `cvt_f32_i32`, for example, converts a signed integer to binary32. The [instruction reference](/reference/instruction-set/#conversions) lists supported pairs.

### 11. Special registers

Sixteen special registers expose thread, wave, workgroup, and grid identity and dimensions. They include lane index and wave width. Programs read them through `mov_sr`.

## What was excluded

The mandatory compute core does not imply support for every target feature. The specification distinguishes optional capabilities from required operations.

### Tensor/matrix operations

Matrix acceleration varies by format, tile shape, and target. WAVE defines optional matrix operations; these require separate capability and lowering support. Scalar arithmetic is not a performance-equivalent substitute for dedicated matrix hardware.

### Texture sampling

Texture sampling includes descriptors, filtering, and coordinate rules outside the core's explicit byte-addressed memory operations.

### Ray tracing

Ray traversal and intersection acceleration are outside the compute core described here.

### Asynchronous copy

An asynchronous copy has completion and ordering requirements beyond an ordinary load/store sequence. Support must account for those requirements and the target's capabilities.

### Variable wave width

Wave width is a parameter. A program that explicitly communicates among 32 lanes still requires that grouping to be preserved on another target. Changing width does not authorize changing a collective's result.

## Universal occupancy equation

A register-capacity bound for resident waves is:

```text
O = floor(F / (R * W * w))
```

`F` is register storage in bytes, `R` is machine registers per lane, `W` is lanes per wave, and `w` is bytes per register. Allocation granularity, workgroup placement, local memory, and hardware limits can lower actual occupancy. WAVE register indices alone do not determine it. See [Occupancy equation](/internals/occupancy-equation/).
