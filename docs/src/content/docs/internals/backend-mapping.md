---
title: Backend Mapping
description: How each WAVE primitive maps to native instructions on NVIDIA, AMD, Intel, and Apple GPUs.
---

Every WAVE primitive has a direct mapping to vendor-native instructions, validating the ISA design principle that portable abstractions should not require emulation on any target.

## Execution Model

| WAVE Concept | NVIDIA PTX | AMD RDNA/CDNA | Intel Xe | Apple Metal |
|-------------|-----------|---------------|----------|-------------|
| Wave | Warp (32 threads) | Wavefront (32 or 64 threads) | Sub-group (8 or 16 threads) | SIMD-group (32 threads) |
| Workgroup | CTA (Cooperative Thread Array) | Workgroup | Workgroup | Threadgroup |
| Thread | Thread / Lane | Work-item / Lane | Channel | Thread / Lane |

The terminology varies but the hierarchy is identical: threads are grouped into waves, waves are grouped into workgroups, workgroups are dispatched across compute units.

## Divergence and Control Flow

| WAVE | NVIDIA PTX | AMD RDNA/CDNA | Intel Xe | Apple Metal |
|------|-----------|---------------|----------|-------------|
| `if` / `else` / `endif` | `@p bra` with convergence barriers; hardware per-thread PC tracks active lanes | `s_cbranch_execz` / `s_cbranch_execnz`; compiler manages `EXEC` mask via `s_and_b64`, `s_or_b64` | Predicated SIMD execution; channel enable mask updated per branch | Hardware divergence stack in `r0l`; structured constructs map nearly 1:1 |
| `loop` / `break` / `endloop` | `bra` with loop-back; `bar.sync` for convergence | `s_branch` / `s_cbranch`; `EXEC` mask tracks active lanes | Predicated loop with channel mask | Hardware loop construct with automatic mask management |
| Predicate registers | Predicate registers (`@p`, `@!p`) | `SCC` (scalar condition) / `VCC` (vector condition) | Flag register + channel enable | Condition code + SIMD mask |

### Key insight

Despite radically different mechanisms (hardware PC tracking vs. compiler-managed masks vs. predicated SIMD vs. hardware stack), all four vendors implement the same observable behavior: lanes that fail a condition are masked off, and all lanes reconverge at the end of the structured region. WAVE specifies the behavior; backends choose the mechanism.

## Memory Operations

| WAVE | NVIDIA PTX | AMD RDNA/CDNA | Intel Xe | Apple Metal |
|------|-----------|---------------|----------|-------------|
| `load.local` | `ld.shared` | `ds_read_b32` (LDS) | `load.slm` (SLM) | `threadgroup` load |
| `store.local` | `st.shared` | `ds_write_b32` (LDS) | `store.slm` (SLM) | `threadgroup` store |
| `load.global` | `ld.global` | `buffer_load_b32` / `flat_load_b32` | `load.ugm` (A64) | `device [[buffer]]` load |
| `store.global` | `st.global` | `buffer_store_b32` / `flat_store_b32` | `store.ugm` (A64) | `device [[buffer]]` store |

### Local memory

All four vendors provide a software-managed scratchpad shared within a workgroup. The names differ (shared / LDS / SLM / threadgroup) but the semantics are identical: fast, low-latency memory visible to all waves in the workgroup, explicitly allocated and addressed.

### Device memory

Global memory access goes through different paths on each vendor (NVIDIA's load/store units, AMD's VMEM pipeline, Intel's LSC, Apple's memory controller) but the programmer-visible behavior is the same: loads and stores to a flat address space visible to all waves on the device.

## Synchronization

| WAVE | NVIDIA PTX | AMD RDNA/CDNA | Intel Xe | Apple Metal |
|------|-----------|---------------|----------|-------------|
| `barrier` | `bar.sync` | `s_barrier` | Gateway barrier | `threadgroup_barrier()` |
| `fence.workgroup` | `membar.cta` | `s_waitcnt lgkmcnt(0)` | `fence.slm` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `fence.device` | `membar.gpu` | `s_waitcnt vmcnt(0)` | `fence.ugm` | `threadgroup_barrier(mem_flags::mem_device)` |
| `fence.system` | `membar.sys` | `s_waitcnt` + cache invalidation | `fence.ugm.sys` | Not directly exposed (system scope is limited) |

### Barriers

Workgroup barriers are the most straightforward mapping: every vendor provides a single instruction that synchronizes all waves in the workgroup. WAVE's `barrier` instruction maps 1:1 in every case.

### Fences

Memory fences show more variation in mechanism (NVIDIA uses explicit `membar` instructions, AMD counts outstanding operations with `s_waitcnt`, Intel uses typed fences, Apple piggybacks on barrier flags) but the semantics align: ensure that memory operations before the fence are visible at the specified scope.

## Atomic Operations

| WAVE | NVIDIA PTX | AMD RDNA/CDNA | Intel Xe | Apple Metal |
|------|-----------|---------------|----------|-------------|
| `atom.add` | `atom.global.add` | `flat_atomic_add` / `global_atomic_add` | `lsc_atomic_iadd` | `atomic_fetch_add_explicit` |
| `atom.cas` | `atom.global.cas` | `flat_atomic_cmpswap` / `global_atomic_cmpswap` | `lsc_atomic_cmpxchg` | `atomic_compare_exchange_weak_explicit` |
| `atom.min` | `atom.global.min` | `flat_atomic_smin` / `global_atomic_smin` | `lsc_atomic_smin` | `atomic_fetch_min_explicit` |
| `atom.max` | `atom.global.max` | `flat_atomic_smax` / `global_atomic_smax` | `lsc_atomic_smax` | `atomic_fetch_max_explicit` |
| `atom.and` | `atom.global.and` | `flat_atomic_and` / `global_atomic_and` | `lsc_atomic_and` | `atomic_fetch_and_explicit` |
| `atom.or` | `atom.global.or` | `flat_atomic_or` / `global_atomic_or` | `lsc_atomic_or` | `atomic_fetch_or_explicit` |
| `atom.xor` | `atom.global.xor` | `flat_atomic_xor` / `global_atomic_xor` | `lsc_atomic_xor` | `atomic_fetch_xor_explicit` |

Every atomic operation in WAVE has a direct 1:1 mapping on all four targets. No emulation is needed.

## Cross-Lane Communication

| WAVE | NVIDIA PTX | AMD RDNA/CDNA | Intel Xe | Apple Metal |
|------|-----------|---------------|----------|-------------|
| `shuffle` | `shfl.sync.idx` | `ds_permute_b32` | `sub_group_shuffle` | `simd_shuffle` |
| `shuffle_down` | `shfl.sync.down` | DPP row shift right | `sub_group_shuffle_down` | `simd_shuffle_down` |
| `shuffle_up` | `shfl.sync.up` | DPP row shift left | `sub_group_shuffle_up` | `simd_shuffle_up` |
| `shuffle_xor` | `shfl.sync.bfly` | DPP butterfly | `sub_group_shuffle_xor` | `simd_shuffle_xor` |

See [Shuffle Primitive](/internals/shuffle-primitive/) for the performance analysis that motivated making shuffle mandatory.

## Mapping Quality

The table below summarizes the mapping quality for each primitive category:

| Category | Mapping type | Backend complexity |
|----------|-------------|-------------------|
| Arithmetic (iadd, fadd, fmul, ...) | 1:1 instruction | Trivial (opcode substitution) |
| Memory (load, store) | 1:1 instruction | Low (address mode translation) |
| Atomics | 1:1 instruction | Low (scope annotation) |
| Barriers | 1:1 instruction | Trivial |
| Fences | 1:1 or short sequence | Low (scope-to-mechanism mapping) |
| Control flow | Short instruction sequence | Medium (mask management varies) |
| Shuffle | 1:1 instruction | Low (variant selection) |

No WAVE primitive requires more than a short instruction sequence on any target. This is by design: every primitive was selected because it exists in hardware on all four vendors (see [Cross-Vendor Analysis](/internals/cross-vendor-analysis/)). The backend is a thin translation layer, not a compiler.
