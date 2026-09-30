---
title: "Cross-vendor analysis"
description: "Common GPU operations, target parameters, and capability differences."
---

WAVE compares NVIDIA, AMD, Intel, and Apple execution interfaces to identify operations with common behavior.

## Source material

| Vendor | Documents | Scope |
|---|---|---|
| NVIDIA | PTX ISA v1.0 through v9.2 | Tesla through Hopper |
| AMD | RDNA 1-4 and CDNA 1-4 ISA references | Consumer and datacenter GPUs |
| Intel | Gen11, Xe-LP, Xe-HPG, and Xe-HPC references | Integrated and discrete GPUs |
| Apple | Metal interfaces and Asahi Linux reverse engineering of G13 | Language guarantees and machine behavior |

Metal describes a programming interface. Asahi Linux documents reverse-engineered hardware behavior. Claims about either must distinguish those sources.

## Analytical framework

The comparison uses eight dimensions.

### 1. Execution model

Threads form execution groups. WAVE represents these as waves without prescribing how hardware schedules their lanes.

### 2. SIMD structure

Wave width varies by target. NVIDIA uses 32 lanes; AMD supports 32 or 64; Intel supports several subgroup widths; Metal exposes the SIMD-group width. Programs must preserve the grouping required by their collectives.

### 3. Register file

Targets differ in register organization, including scalar and vector files and SIMD register widths. WAVE uses untyped 32-bit lane values as its common representation. A backend maps that state to the target compiler's registers.

### 4. Memory hierarchy

Workgroup-local and device memory provide common visibility domains. Cache organization and coherence mechanisms remain target-specific.

### 5. Instruction categories

Arithmetic, memory access, atomics, control flow, synchronization, and lane exchange occur across the target interfaces. Operand formats and participation rules still require separate comparison.

### 6. Synchronization

Barriers coordinate arrival, fences order accesses, and atomics update memory indivisibly. Their scopes and failure behavior matter as much as their names.

### 7. Scheduling

The target controls instruction issue and resource assignment. WAVE specifies observable program behavior without exposing a particular hardware scheduler.

### 8. Design choices

Matrix acceleration, wave-width modes, subgroup sizes, and graphics integration vary. Optional capabilities need explicit requirements rather than assumptions about all targets.

## Classification of findings

### Invariants: directly adopted by WAVE

The core represents lane state, memory accesses, structured control flow, synchronization, and wave exchange. Each operation defines its participating lanes and observable effects.

### Parameterizable dialects: exposed as runtime queries

Wave width, register limits, local-memory capacity, and maximum workgroup dimensions vary by target. Programs and launch code must account for those limits. A width change can alter a collective and requires compatible lowering. See [Thin abstraction](/internals/thin-abstraction/).

### True divergences: excluded from WAVE

Machine encodings, cache organization, and scheduling mechanisms belong to the target implementation. Specialized operations require capabilities beyond the mandatory core. Matrix operations, for example, have optional WAVE instructions rather than a universal machine-level mapping.

<span id="key-insight-the-8020-rule"></span>

## Shared behavior and execution cost

The ISA defines shared execution effects while backends select target-specific mechanisms. Performance depends on the resulting programs and hardware.
