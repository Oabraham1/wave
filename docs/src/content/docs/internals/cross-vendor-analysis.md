---
title: Cross-Vendor Analysis
description: How analysis of 5,000+ pages of GPU documentation across four vendors shaped WAVE's design decisions.
---

WAVE's instruction set and execution model emerged from a systematic analysis of GPU architectures from NVIDIA, AMD, Intel, and Apple, covering over 5,000 pages of vendor documentation.

## Source Material

| Vendor | Documents | Scope |
|--------|-----------|-------|
| **NVIDIA** | PTX ISA v1.0 through v9.2 | 20+ years of ISA evolution, from Tesla to Hopper |
| **AMD** | RDNA 1--4, CDNA 1--4 ISA reference guides | Consumer and datacenter GPU architectures |
| **Intel** | Gen11, Xe-LP, Xe-HPG, Xe-HPC programmer's reference manuals | Integrated and discrete GPU generations |
| **Apple** | G13 ISA (reverse-engineered via Asahi Linux project) | Apple Silicon GPU architecture |

The Apple documentation is notably different from the others: Apple does not publish a public ISA reference. The analysis relied on reverse-engineering work by the Asahi Linux project, cross-referenced with Metal shader compiler output and Metal Performance Shaders behavior.

## Analytical Framework

Each vendor's architecture was analyzed along 8 dimensions:

### 1. Execution Model

How threads are grouped, scheduled, and retired.

**Finding**: All four vendors group threads into fixed-width SIMD groups (waves). The grouping is mandatory and hardware-enforced. No vendor supports fully independent thread execution at the compute-unit level.

### 2. SIMD Structure

The width and organization of SIMD execution.

**Finding**: Wave widths vary (NVIDIA 32, AMD 32 or 64, Intel 8 or 16, Apple 32), but the concept of a fixed-width lockstep group is universal. The width is a parameter, not a design choice that WAVE should fix.

### 3. Register File

Size, width, typing, and allocation strategy.

**Finding**: All four vendors use untyped 32-bit register files. Sizes and per-thread maximums vary, but the fundamental model --- large flat register file, untyped, partitioned across threads --- is invariant.

### 4. Memory Hierarchy

Levels, coherence domains, and addressing modes.

**Finding**: All four vendors provide at least local (workgroup-shared) memory and device (global) memory. Cache hierarchies vary significantly, but the programmer-visible memory spaces are consistent. Scoped memory ordering (not TSO) is universal.

### 5. Instruction Categories

What operations the hardware supports natively.

**Finding**: Arithmetic (integer, floating-point), memory (load, store, atomic), control flow (branch, barrier), and communication (shuffle) operations are present on all four architectures. The specific instruction encodings differ, but the functional categories are identical.

### 6. Synchronization

Barriers, fences, and atomic operations.

**Finding**: All vendors provide workgroup barriers, memory fences at multiple scopes, and atomic operations on global memory. The mechanisms differ (NVIDIA uses `bar.sync`, AMD uses `s_barrier`, Intel uses gateway barriers, Apple uses `threadgroup_barrier`), but the semantics are equivalent.

### 7. Scheduling

How waves are assigned to execution resources.

**Finding**: This is where vendors diverge most significantly. NVIDIA uses hardware warp schedulers with scoreboarding. AMD uses a mix of hardware and compiler-managed scheduling. Intel uses a thread dispatcher. Apple uses a hardware scheduler with different occupancy characteristics. Scheduling is an implementation detail that WAVE deliberately does not expose.

### 8. Design Choices

Vendor-specific features that reflect architectural philosophy.

**Finding**: Each vendor makes unique trade-offs (NVIDIA's tensor cores, AMD's wave32/wave64 modes, Intel's variable sub-group sizes, Apple's tile-based rendering integration). These are vendor-differentiating features, not candidates for a portable ISA.

## Classification of Findings

The analysis produced three categories:

### Invariants (directly adopted by WAVE)

Features present on all four architectures in essentially the same form:

- Wave-based execution (threads grouped into fixed-width SIMD groups)
- Untyped 32-bit registers
- Local (workgroup-shared) memory
- Device (global) memory
- Structured control flow with divergence management
- Workgroup barriers
- Scoped memory fences
- Atomic operations
- Shuffle (cross-lane data exchange)

These became WAVE's mandatory primitive set. If all four vendors implement something, it imposes no additional hardware burden and omitting it would leave performance on the table.

### Parameterizable Dialects (exposed as runtime queries)

Features present on all architectures but with different concrete values:

- Wave width (32, 64, 8, 16)
- Maximum registers per thread (255, 256, 128, 128)
- Register file size (256KB, 512KB, 128KB, 208KB)
- Local memory size per workgroup
- Maximum workgroup dimensions

These are not abstracted away --- they are exposed through WAVE's query interface so that programs can adapt. See [Thin Abstraction](/internals/thin-abstraction/).

### True Divergences (excluded from WAVE)

Features that differ fundamentally across vendors and cannot be meaningfully unified:

- Scheduling policy (hardware vs. compiler-managed vs. hybrid)
- Cache hierarchy details (L1/L2 sizes, replacement policies, coherence protocols)
- Vendor-specific extensions (tensor cores, ray tracing units, display controllers)
- ISA-specific encoding details (instruction formats, register naming conventions)

These are left to the backend. WAVE does not attempt to abstract ray tracing, tensor operations, or cache control hints --- these would either constrain the abstraction to the least capable vendor or require extensive feature detection that undermines portability.

## Key Insight: The 80/20 Rule

Approximately 80% of GPU compute functionality is identical across vendors. The remaining 20% accounts for nearly all the performance differentiation. WAVE targets the 80% --- the common substrate that every GPU provides --- and leaves the 20% to vendor-specific backends and extensions.

This is not a compromise. The 80% covers the complete set of operations needed for general-purpose GPU computing: arithmetic, memory access, synchronization, and communication. The 20% covers specialized accelerators and microarchitectural optimizations that change with every hardware generation.
