---
title: Memory Scoping
description: Why WAVE uses scoped acquire/release memory ordering instead of TSO or fully relaxed semantics.
---

WAVE adopts a scoped acquire/release memory model that threads the needle between TSO (too restrictive for GPU hardware) and fully relaxed semantics (too weak for programmers to reason about correctly).

## The Memory Model Spectrum

GPU memory models sit on a spectrum:

- **TSO (Total Store Order)**: Used by x86 CPUs. Every store is immediately visible to all threads in program order. Simple to reason about, but GPUs cannot implement it efficiently --- their deep memory hierarchies, per-CU caches, and asynchronous memory controllers would require constant flushing.

- **Fully relaxed**: Loads and stores can be reordered arbitrarily. Maximizes hardware freedom but pushes all correctness burden onto programmers, who must insert fences everywhere.

- **Scoped acquire/release**: The middle ground. Ordering is enforced only when explicitly requested, and only at the scope where it matters.

## Four Scope Levels

WAVE defines four memory scopes, matching the GPU execution hierarchy:

| Scope | Visibility | Use case |
|-------|-----------|----------|
| **Wave** | Lanes within the executing wave | Shuffle results, wave-level reductions |
| **Workgroup** | All waves in the workgroup | Shared local memory synchronization |
| **Device** | All waves on the GPU | Global memory coordination between workgroups |
| **System** | GPU and host CPU | Host-device communication, mapped buffers |

### Why four levels?

Each scope corresponds to a distinct hardware boundary with different latency and coherence characteristics:

- **Wave scope** is free --- lanes within a wave share a register file or L0 cache. No fence is needed.
- **Workgroup scope** requires flushing to the local data store (LDS/SLM/threadgroup memory), which is fast (tens of cycles).
- **Device scope** requires flushing to L2 or device memory, which is slower (hundreds of cycles).
- **System scope** requires cache invalidation visible to the CPU's coherence domain, which is the most expensive (potentially thousands of cycles).

Collapsing these into fewer levels would force over-synchronization. For example, if workgroup and device scope were merged, every local memory fence would pay the cost of a global fence.

## Acquire/Release Semantics

WAVE uses acquire/release ordering for atomic operations and fences:

- **Acquire**: No subsequent memory operation in the executing thread can be reordered before this operation. Used when reading shared state ("I need to see everything that happened before the release that made this value visible").

- **Release**: No preceding memory operation in the executing thread can be reordered after this operation. Used when publishing shared state ("everything I wrote before this point must be visible to anyone who acquires").

```
; Producer (wave 0)
store.global [addr], r0            ; write data
fence.release.workgroup            ; ensure store is visible at workgroup scope
store.global.release [flag], r1    ; signal completion

; Consumer (wave 1)
load.global.acquire r2, [flag]     ; acquire: see all stores before the release
load.global [addr], r3             ; guaranteed to see r0's value
```

### Scoped fences

The `scope` field in the instruction encoding (bits [6:5]) specifies the fence scope. A `fence.release.workgroup` ensures visibility only within the workgroup --- it does not flush to device-wide caches. A `fence.release.device` flushes further. The backend maps each scope to the appropriate vendor-specific mechanism.

## Vendor Memory Model Mapping

| Concept | NVIDIA | AMD | Intel | Apple |
|---------|--------|-----|-------|-------|
| **Formal model** | Axiomatic (published with PTX) | Operational (`S_WAITCNT` counters) | Scoreboard-based | Implicit (async loads) |
| **Workgroup fence** | `membar.cta` | `s_waitcnt lgkmcnt(0)` | `fence.slm` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| **Device fence** | `membar.gpu` | `s_waitcnt vmcnt(0)` | `fence.ugm` | `threadgroup_barrier(mem_flags::mem_device)` |
| **System fence** | `membar.sys` | `s_waitcnt vmcnt(0)` + cache flush | `fence.ugm.sys` | Not directly exposed |
| **Acquire** | `ld.acquire` | `s_waitcnt` + `buffer_gl0_inv` | `load.ugm.ca` | Implicit via load ordering |
| **Release** | `st.release` | `s_waitcnt` before store | `store.ugm.uc` | Implicit via store ordering |

### Key observation

Every vendor provides mechanisms that map naturally to scoped acquire/release. No vendor requires TSO. No vendor leaves ordering fully relaxed at the ISA level --- they all provide some form of scoped fencing. WAVE's model captures the common denominator.

## Why Not Relaxed + Fences Everywhere?

A fully relaxed model with explicit fences would technically work, but it shifts the correctness burden entirely to the programmer (or compiler). Experience from C11/C++11 memory models shows that relaxed atomics are a persistent source of bugs. By making acquire/release the default for atomic operations and providing scoped fences, WAVE ensures that the most common synchronization patterns are correct by construction.

## Why Not TSO?

TSO would require that every store become globally visible before any subsequent load executes. On a GPU with 64+ active waves per compute unit, each with 32--64 lanes, this would require serializing thousands of concurrent stores through a single coherence point. The performance cost would eliminate the GPU's primary advantage: massive parallelism. Every vendor's hardware is designed around relaxed ordering with explicit synchronization points, and WAVE's model respects this reality.
