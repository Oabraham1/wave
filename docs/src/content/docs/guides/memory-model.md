---
title: Memory Model
description: Understand WAVE's three memory levels, load/store operations, cache hints, atomics, fences, and memory scopes.
---

WAVE exposes three distinct memory levels - registers, local memory, and device memory - each with different scope, latency, and capacity trade-offs.

## Memory Hierarchy at a Glance

```
┌─────────────────────────────────────────────┐
│              Device Memory                  │
│         (global, GB scale, ~300 cycles)     │
├─────────────────────────────────────────────┤
│           Local Memory (per-workgroup)      │
│       (shared, 48–96 KB, ~20 cycles)       │
├──────────┬──────────┬──────────┬────────────┤
│  Thread  │  Thread  │  Thread  │   ...      │
│  r0–r31  │  r0–r31  │  r0–r31  │            │
│  (1 cyc) │  (1 cyc) │  (1 cyc) │            │
└──────────┴──────────┴──────────┴────────────┘
```

| Level | Scope | Addressable | Typical Size | Latency |
|---|---|---|---|---|
| Registers (`r0`–`r31`) | Per-thread | By name | 32 x 32-bit per thread | ~1 cycle |
| Local memory | Per-workgroup | Byte-addressable | 48–96 KB | ~20 cycles |
| Device memory | Global | 64-bit byte-addressable | GBs | ~300 cycles |

## Registers

Each thread has its own set of 32 general-purpose registers (`r0`–`r31`). Registers are the fastest storage but the scarcest resource. Other threads cannot see your registers - they are completely private.

```asm
mov_imm r0, 100       ; immediate to register
iadd    r2, r0, r1    ; register-to-register arithmetic
```

## Load and Store Operations

### Widths

Both `local_load`/`local_store` and `device_load`/`device_store` support five widths:

| Suffix | Width | Use case |
|---|---|---|
| `u8` | 8-bit | Byte-level data, characters |
| `u16` | 16-bit | Half-precision floats, short integers |
| `u32` | 32-bit | Standard floats and integers |
| `u64` | 64-bit | Doubles, 64-bit addresses, pointers |
| `u128` | 128-bit | Wide loads for throughput (4 packed floats) |

```asm
device_load.u32  r0, r1      ; load 32 bits from device address in r1
device_load.u128 r0, r1      ; load 128 bits (fills r0–r3)
local_load.u32   r0, r1      ; load 32 bits from local address in r1
device_store.u64 r0, r1      ; store 64 bits (r0–r1) to device address
```

### Cache Hints

Every device memory operation accepts an optional cache hint that controls caching behavior:

| Hint | Value | Meaning |
|---|---|---|
| `cached` | 0 | Default. Use the cache hierarchy normally. |
| `uncached` | 1 | Bypass caches. Use for data that will not be reused. |
| `streaming` | 2 | Cache with low priority. Use for data accessed once in a streaming pattern. |

```asm
device_load.u32.cached    r0, r1   ; normal caching (default)
device_load.u32.uncached  r0, r1   ; bypass cache
device_load.u32.streaming r0, r1   ; streaming / low-priority cache
```

**When to use each hint:**
- Use `cached` (or omit the hint) for data that may be read multiple times.
- Use `streaming` for large sequential scans where each element is touched once.
- Use `uncached` for data written by other workgroups that you need to read fresh.

## Local Memory (Shared Memory)

Local memory is shared among all threads in a workgroup. It is small (48–96 KB), fast (~20 cycle latency), and byte-addressable. Common uses include:

- Staging data from device memory for repeated access
- Communication between threads in the same workgroup
- Building partial results (e.g., reduction scratch space)

```asm
; Thread 0 writes a value; all threads in the workgroup can read it.
mov_imm r0, 42
local_store.u32 r0, r1          ; store r0 at local address r1

barrier                          ; ensure all threads see the write

local_load.u32 r2, r1           ; any thread can now load the value
```

**Important:** You must insert a `barrier` between a local store and a local load from a different thread. Without the barrier, a thread may read stale or uninitialized data.

### Example: Cooperative Device-to-Local Tiling

A common pattern is for each thread to load one element from device memory into local memory, synchronize, then have every thread read from the faster local copy:

```asm
; Each thread loads one element from device memory into local memory
mov_sr  r0, sr_thread_id_x
shl     r1, r0, 2               ; local byte offset = tid * 4
iadd    r2, r10, r1             ; device address = base + offset
device_load.u32 r3, r2          ; load from device memory
local_store.u32 r3, r1          ; store into local memory

barrier                          ; wait for all threads to finish writing

; Now every thread can read any element from local memory
; e.g., read the neighbor's value
iadd    r4, r0, 1               ; neighbor index = tid + 1
shl     r5, r4, 2               ; neighbor byte offset
local_load.u32 r6, r5           ; load neighbor's value from local memory
```

## Atomic Operations

Atomics perform read-modify-write on a single memory location indivisibly. They work on both local and device memory.

### Available Atomic Operations

| Operation | Syntax | Description |
|---|---|---|
| Add | `atomic_add` | `*addr += val` |
| Sub | `atomic_sub` | `*addr -= val` |
| Min | `atomic_min` | `*addr = min(*addr, val)` |
| Max | `atomic_max` | `*addr = max(*addr, val)` |
| And | `atomic_and` | `*addr &= val` |
| Or | `atomic_or` | `*addr \|= val` |
| Xor | `atomic_xor` | `*addr ^= val` |
| Exchange | `atomic_exchange` | `old = *addr; *addr = val; return old` |
| CAS | `atomic_cas` | Compare-and-swap |

```asm
; Atomically add r0 to the value at device address r1
; Result (old value) returned in r2
atomic_add r2, r1, r0

; Compare-and-swap: if *r1 == r0, set *r1 = r3; old value in r2
atomic_cas r2, r1, r0, r3
```

## Memory Scopes

Every atomic and fence operation has a **scope** that determines which threads are guaranteed to observe the effect:

| Scope | Value | Visibility |
|---|---|---|
| `wave` | 0 | Threads within the same wave |
| `workgroup` | 1 | Threads within the same workgroup |
| `device` | 2 | All threads on the GPU |
| `system` | 3 | GPU and CPU (host) |

Choose the narrowest scope that satisfies your correctness requirements. Wider scopes are more expensive because the hardware must flush or invalidate more caches.

## Fences

Fences enforce ordering of memory operations without operating on a specific address. WAVE provides three fence types:

| Fence | Guarantees |
|---|---|
| `fence_acquire` | All loads after this fence see writes that happened before a matching release. |
| `fence_release` | All writes before this fence are visible to threads that perform a matching acquire. |
| `fence_acq_rel` | Both acquire and release semantics combined. |

Each fence takes a scope:

```asm
fence_release.workgroup     ; make all prior writes visible within the workgroup
barrier                      ; synchronize threads
fence_acquire.workgroup     ; see all writes from before the barrier
```

### When to Use Each Scope

- **`wave`**: Communication between lanes in the same wave (usually handled by wave ops instead).
- **`workgroup`**: The most common scope. Use with `barrier` for local memory synchronization.
- **`device`**: Cross-workgroup communication through device memory (e.g., global counters, producer-consumer between workgroups).
- **`system`**: When the CPU needs to observe GPU writes, or vice versa (e.g., signaling completion to the host).

### Example: Device-Scope Counter

```asm
; Atomically increment a global counter visible to all workgroups
mov_imm r0, 1
atomic_add r1, r10, r0      ; r10 = address of global counter

; Ensure subsequent reads see the updated counter across the device
fence_acq_rel.device
```

## Barrier

`barrier` is a workgroup-level synchronization point. When a thread reaches a barrier, it waits until every thread in the workgroup has also reached it. This is the standard way to synchronize local memory access.

**Rule of thumb:** if one thread writes to local memory and another thread reads that address, there must be a `barrier` between the write and the read.

```asm
local_store.u32 r0, r1      ; write
barrier                       ; all threads sync here
local_load.u32 r2, r3        ; safe to read any thread's write
```

## Summary

| Operation | Local Memory | Device Memory |
|---|---|---|
| Load | `local_load.{u8–u128}` | `device_load.{u8–u128}[.hint]` |
| Store | `local_store.{u8–u128}` | `device_store.{u8–u128}[.hint]` |
| Atomics | `atomic_*` on local addr | `atomic_*` on device addr |
| Synchronization | `barrier` + fences | Fences with `device`/`system` scope |

Next: [Control Flow](/guides/control-flow/) - learn how branching, loops, and divergence work in WAVE assembly.
