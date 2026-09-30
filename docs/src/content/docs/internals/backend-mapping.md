---
title: "Backend mapping"
description: "How WAVE operations map to PTX, HIP, SYCL, and Metal."
---

Backends translate WAVE instructions into vendor compiler inputs. The vendor compiler selects machine instructions. One WAVE instruction can require several target operations.

## Execution model

| WAVE concept | NVIDIA PTX | AMD HIP | Intel SYCL | Apple Metal |
|---|---|---|---|---|
| Wave | Warp, 32 lanes | Wavefront, 32 or 64 lanes | Sub-group with target-dependent width | SIMD-group |
| Workgroup | CTA | Thread block | Workgroup | Threadgroup |
| Thread | Thread | Thread | Work-item | Thread |

Threads belong to waves and workgroups. The target determines how those groups use its execution resources.

## Divergence and control flow

PTX uses predicates and branch labels. HIP, SYCL, and Metal backends emit structured source control flow. WAVE defines active-lane behavior; vendor compilers implement the target's branch and reconvergence mechanism.

<span id="key-insight"></span>

### Active lanes

A translation must preserve which lanes execute each operation and which lanes remain live at reconvergence.

## Memory operations

| WAVE memory | PTX | HIP | SYCL | Metal |
|---|---|---|---|---|
| Local | `.shared` | Dynamic shared allocation | `local_accessor` | `threadgroup` allocation |
| Device | `ld.global` and `st.global` | Device pointer | USM pointer | `device` buffer pointer |

### Local memory

Each workgroup owns a separate local-memory region. Its size comes from kernel metadata. Threads use explicit synchronization when exchanging values through it.

### Device memory

Backends load and store values through the device-memory base pointer and WAVE byte offsets. The launch code must supply an allocation large enough for those accesses.

## Synchronization

| WAVE operation | PTX | HIP | SYCL | Metal |
|---|---|---|---|---|
| Workgroup barrier | `bar.sync 0` | `__syncthreads` | `group_barrier` | `threadgroup_barrier` |
| Fence | `membar` with scope | `__threadfence` variants | `atomic_fence` | `atomic_thread_fence` |

### Barriers

A barrier coordinates arrival within a workgroup. Programs must satisfy its participation requirements.

### Fences

A fence orders memory accesses without a rendezvous. Metal supports wave, workgroup, and device scopes through `atomic_thread_fence`; it rejects system scope. See [Memory scoping](/internals/memory-scoping/) for the scope mappings.

## Atomic operations

| Operation | PTX | HIP | SYCL | Metal |
|---|---|---|---|---|
| Add | `atom.add` | `atomicAdd` | `atomic_ref::fetch_add` | `atomic_fetch_add_explicit` |
| Compare-and-swap | `atom.cas` | `atomicCAS` | Compare/exchange on `atomic_ref` | Weak compare/exchange with retry |
| Minimum and maximum | `atom.min`, `atom.max` | `atomicMin`, `atomicMax` | Minimum/maximum on `atomic_ref` | Atomic minimum/maximum functions |
| Bitwise operations | `atom.and`, `atom.or`, `atom.xor` | `atomicAnd`, `atomicOr`, `atomicXor` | Bitwise operations on `atomic_ref` | Atomic bitwise functions |

The translation must preserve return values and comparison operands as well as the memory update. Metal retries a weak compare/exchange when it fails spuriously. PTX subtraction negates the operand before atomic addition.

## Cross-lane communication

| WAVE operation | PTX | HIP | SYCL | Metal |
|---|---|---|---|---|
| Shuffle | `shfl.sync.idx` | `__shfl` | `select_from_group` | `simd_shuffle` |
| Shuffle down | `shfl.sync.down` | `__shfl_down` | `shift_group_left` | `simd_shuffle_down` |
| Shuffle up | `shfl.sync.up` | `__shfl_up` | `shift_group_right` | `simd_shuffle_up` |
| Shuffle XOR | `shfl.sync.bfly` | `__shfl_xor` | `permute_group_by_xor` | `simd_shuffle_xor` |

Source lanes and participating lanes must satisfy the target's collective-operation requirements. See [Shuffle primitive](/internals/shuffle-primitive/).

## Mapping quality

Arithmetic, loads, stores, and shuffles often map to corresponding target operations. Control flow, conversions, and atomics can require sequences. Register allocation, instruction selection, and scheduling occur in the vendor compiler, so generated source alone does not establish execution cost.
