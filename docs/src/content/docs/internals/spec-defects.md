---
title: Spec Defects
description: Three specification defects found during WAVE development, their root causes, and how each was resolved.
---

The WAVE specification has undergone three significant corrections since v0.1, each discovered through systematic verification against hardware behavior and encoding constraints.

## Defect 1: Register Encoding Mismatch (v0.1)

### Symptom

The v0.1 specification text described 256 general-purpose registers per thread, but the instruction encoding used 5-bit register fields.

### Root cause

Five bits encode values 0--31, addressing only 32 registers. To address 256 registers, the encoding would require 8-bit register fields (3 fields x 8 bits = 24 bits for registers alone, leaving only 8 bits for opcode, modifier, scope, predicate, and flags in a 32-bit word --- clearly insufficient).

The spec text was written aspirationally ("256 registers, like NVIDIA and AMD support"), while the encoding was designed pragmatically. The two were never reconciled before publication.

### Fix

The register count was reduced to 32 to match the 5-bit encoding. This was validated as sufficient by analyzing register pressure across compute kernels from public benchmark suites. Thirty-two registers cover the vast majority of kernels without spilling; the small number of outliers are handled by the compiler's spill-to-local-memory mechanism.

### Lesson

Encoding and specification text must be verified against each other mechanically, not just reviewed by humans. A simple check --- "does the bit width support the claimed range?" --- would have caught this immediately.

## Defect 2: Shared Divergence Stack (v0.1)

### Symptom

Programs with multiple waves per workgroup could deadlock when waves reached barriers at different points in divergent control flow.

### Root cause

The v0.1 specification defined a single divergence stack shared across all waves in a workgroup. The intended design was to save hardware resources by sharing the stack. The actual effect was cross-wave state corruption.

Consider two waves, A and B, in the same workgroup:

1. **Cycle N**: Wave A enters an `if` block, pushing its active mask onto the shared stack.
2. **Cycle N+1**: Wave B, executing a different part of the program, enters a different `if` block and pushes its own mask onto the same stack.
3. **Cycle N+2**: Wave A reaches `endif` and pops the stack --- but it pops Wave B's mask, not its own.
4. **Result**: Wave A's active mask is corrupted. If either wave then hits a barrier, the corrupted masks prevent proper synchronization, causing deadlock.

This defect was not theoretical. It was triggered by the emulator (`wave-emu`) when running a parallel reduction kernel with 4 waves per workgroup.

### Fix

Each wave gets its own divergence stack. This matches the hardware implementation of all four target vendors:

- NVIDIA: per-warp convergence state.
- AMD: per-wavefront EXEC mask stack (managed by the compiler).
- Intel: per-sub-group channel enable stack.
- Apple: per-SIMD-group hardware stack in `r0l`.

No vendor shares divergence state across waves, so the shared-stack design was not just buggy --- it was unmappable to any real hardware.

### Lesson

Resource-sharing optimizations in a specification must be validated against every target's execution model. A spec-level optimization that no hardware implements is a red flag.

## Defect 3: Modifier Field Overflow (v0.2)

### Symptom

The assembler could not encode `fsin`, `fcos`, `fexp2`, `flog2`, or atomic compare-and-swap (`atom.cas`) instructions.

### Root cause

The v0.2 encoding allocated 3 bits for the modifier field (bits [10:8]), supporting values 0--7. The `FUnaryOp` opcode class defines 12 variants (0--11), and the atomic opcode class requires modifier value 8 for CAS. Both exceed the 3-bit range.

### Fix

The modifier field was expanded to 4 bits (bits [10:7]) in v0.3, supporting values 0--15. To reclaim the bit, the flags field was reduced from 3 bits to 2 bits by eliminating two flags:

| Eliminated Flag | Replacement |
|----------------|-------------|
| `WAVE_REDUCE_FLAG` | Reductions became distinct opcodes rather than flagged variants of existing operations. |
| `NON_RETURNING_ATOMIC_FLAG` | Atomics that discard their result write to the zero register (`r0`), making the flag implicit. |

Full details in [Modifier Field Evolution](/internals/modifier-field/).

### Lesson

Bit allocation in an instruction encoding must be validated against the complete enumeration of every opcode variant, including uncommon operations like transcendental functions and atomic CAS. The defect was invisible when testing only arithmetic operations and only manifested when the assembler attempted to encode the full ISA.

## Summary

| Defect | Version | Category | Impact | Fix Version |
|--------|---------|----------|--------|-------------|
| Register encoding mismatch | v0.1 | Spec/encoding inconsistency | Misleading documentation | v0.2 |
| Shared divergence stack | v0.1 | Correctness (deadlock) | Programs could hang | v0.2 |
| Modifier field overflow | v0.2 | Encoding capacity | ISA incomplete | v0.3 |

All three defects were caught before any hardware or production backend implementation, validating the approach of building a software emulator (`wave-emu`) and assembler (`wave-compiler`) as specification verification tools.
