---
title: "Memory scoping"
description: "Memory-ordering scopes, fences, and acquire/release semantics."
---

WAVE uses scoped acquire/release ordering. A scope identifies the participating threads; an ordering operation constrains the memory accesses visible to them.

## The memory model spectrum

Ordinary loads and stores do not establish synchronization between threads. Atomics, fences, and barriers provide the ordering required by an algorithm. A backend must preserve those requirements without assuming that its target follows a CPU memory model such as total store order, or TSO.

## Four scope levels

| Scope | Participants | Use case |
|---|---|---|
| Wave | Lanes in the executing wave | Ordering within a wave |
| Workgroup | Threads in the workgroup | Local-memory communication |
| Device | Threads on the device | Communication between workgroups |
| System | Device and host participants | Host-device communication |

### Why four levels?

A narrower scope can use a more local synchronization mechanism. The cost depends on the device, the memory involved, and preceding accesses. Wave scope does not make all memory accesses implicitly ordered. A backend must reject a scope it cannot implement.

## Acquire/release semantics

Acquire orders subsequent accesses after the acquire operation. Release orders preceding accesses before the release operation. Communication between threads also requires a matching synchronization relationship at a scope that includes both participants.

A producer publishes data before a release operation. A consumer must observe that publication through the corresponding acquire operation before reading the data. A fence alone does not prove that the producer has finished.

### Scoped fences

Scoped instructions store their scope in word1 bits [1:0]. The assembler accepts `fence_acquire`, `fence_release`, and `fence_acq_rel` followed by a scope.

```asm
fence_release workgroup
fence_acquire device
fence_acq_rel system
```

These examples show syntax, not a complete producer-consumer protocol.

## Vendor memory model mapping

The backends emit the following fence operations:

| Scope | PTX | HIP | SYCL | Metal |
|---|---|---|---|---|
| Wave | `membar.cta` | `__threadfence_block` | Sub-group scope | SIMD-group scope |
| Workgroup | `membar.cta` | `__threadfence_block` | Workgroup scope | Threadgroup scope |
| Device | `membar.gl` | `__threadfence` | Device scope | Device scope |
| System | `membar.sys` | `__threadfence_system` | System scope | Rejected |

SYCL uses `atomic_fence` with the requested order and scope. Metal uses `atomic_thread_fence` with sequentially consistent ordering and both threadgroup and device memory flags. This stronger ordering preserves the requested acquire/release requirements.

### Key observation

A memory fence orders accesses without requiring threads to rendezvous. A workgroup barrier coordinates arrival as well as memory ordering. Replacing a device fence with a workgroup barrier changes the participating threads and is not a valid scope translation.

## Why not relaxed + fences everywhere?

The instruction contract identifies the required ordering and scope. The programmer or compiler must choose operations that establish the algorithm's synchronization relationship. Acquire/release operations make that relationship explicit; they do not make unsynchronized algorithms correct automatically.

## Why not TSO?

GPU programs coordinate at several scopes and use distinct local and device memory spaces. WAVE expresses those requirements directly rather than requiring a target to implement a CPU ordering model.
