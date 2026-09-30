---
title: "Memory model"
description: "Registers, shared memory, device memory, atomics, and synchronization."
---

WAVE uses registers, local memory, and device memory. Their scope and synchronization requirements differ.

## Memory hierarchy at a glance

| Storage | Scope | Access |
|---|---|---|
| Registers | One thread | Register operands and wave operations |
| Local memory | One workgroup | Byte addresses |
| Device memory | Device allocation | Byte addresses |

Capacity, latency, and cache behavior depend on the target. Declare local memory with `.local_memory` and check the target's resource limits.

## Registers

Registers hold untyped 32-bit values. Instructions determine whether those bits represent integers, floats, or packed values. Threads exchange register values through wave operations.

```asm
mov_imm r0, 100
iadd r2, r0, r1
```

## Load and store operations

### Widths

Device loads and stores support `u8`, `u16`, `u32`, `u64`, and `u128`. Local loads and stores support `u8`, `u16`, `u32`, and `u64`.

Loads name the destination before the address. Stores name the address before the value:

```asm
device_load_u32 r0, r1       ; load from address r1 into r0
device_load_u128 r4, r1      ; load four words into r4 through r7
local_load_u32 r0, r1
device_store_u64 r2, r4      ; store r4 and r5 at address r2
```

Multiword operations use consecutive registers. Keep the accessed range inside the allocation and satisfy the instruction's alignment requirements.

### Cache hints

Device operations accept cache hints:

| Hint | Intent |
|---|---|
| `cached` | Reuse data through the cache hierarchy |
| `uncached` | Request cache bypass |
| `streaming` | Give streaming data lower cache priority |

```asm
device_load_u32.cached r0, r1
device_load_u32.uncached r0, r1
device_load_u32.streaming r0, r1
```

Hints express cache policy. They do not replace atomic operations, fences, or barriers needed for synchronization. A backend maps them to the policies its target supports.

## Local memory: shared memory

Threads in a workgroup share local memory. Use it to exchange values, stage tiles, or store partial results.

```asm
mov_sr r0, sr_thread_id_x
mov_imm r1, 4
imul r2, r0, r1
local_store_u32 r2, r3
barrier
local_load_u32 r4, r2
```

The kernel must declare enough local memory for all participating threads. Insert a barrier before another thread reads the stored values.

### Example: cooperative device-to-local tiling

Each thread loads one word and stores it at its own local offset. After the barrier, threads may read the tile:

```asm
mov_sr r0, sr_thread_id_x
mov_imm r1, 4
imul r2, r0, r1
iadd r3, r10, r2
device_load_u32 r4, r3
local_store_u32 r2, r4
barrier

; Read the next thread's value, wrapping at the workgroup boundary.
mov_imm r5, 1
iadd r6, r0, r5
mov_sr r7, sr_workgroup_size_x
imod r6, r6, r7
imul r6, r6, r1
local_load_u32 r8, r6
```

Here `r10` contains the device tile's base offset. The allocation must contain one word per thread.

## Atomic operations

An atomic operation updates one memory location indivisibly. Local atomics use `local_atomic_*`; device atomics use `atomic_*` with a scope.

### Available atomic operations

| Operation | Device mnemonic |
|---|---|
| Add | `atomic_add` |
| Subtract | `atomic_sub` |
| Minimum | `atomic_min` |
| Maximum | `atomic_max` |
| Bitwise AND | `atomic_and` |
| Bitwise OR | `atomic_or` |
| Bitwise XOR | `atomic_xor` |
| Exchange | `atomic_exchange` |
| Compare and swap | `atomic_cas` |

```asm
atomic_add r2, r1, r0, .device
atomic_cas r2, r1, r0, r3, .device
```

Both examples return the previous value in `r2`. The compare-and-swap stores `r3` only when the value at address `r1` equals `r0`.

## Memory scopes

A scope identifies the participants covered by an atomic operation or fence:

| Scope | Participants |
|---|---|
| `wave` | Lanes in one wave |
| `workgroup` | Threads in one workgroup |
| `device` | Threads on the device |
| `system` | Device and host participants |

Choose the scope required by the communication pattern and supported by the target.

## Fences

Fences order memory operations. They do not wait for other threads to arrive.

| Fence | Ordering |
|---|---|
| `fence_acquire` | Acquire ordering |
| `fence_release` | Release ordering |
| `fence_acq_rel` | Acquire and release ordering |

```asm
fence_release .workgroup
barrier
fence_acquire .workgroup
```

### When to use each scope

- Use `wave` for communication within a wave.
- Use `workgroup` for communication within a workgroup.
- Use `device` for communication between workgroups.
- Use `system` when the synchronization protocol includes the host.

### Example: device-scope counter

```asm
mov_imm r0, 1
atomic_add r1, r10, r0, .device
```

The operation increments the counter at `r10`. Publishing other memory through a counter also requires a suitable ordering protocol; atomicity alone is insufficient.

## Barrier

`barrier` synchronizes workgroup threads. All participating threads must reach the same barrier. Use it between shared-memory writes and reads by other threads, and before overwriting storage that other threads may still be reading.

## Summary

| Operation | Local memory | Device memory |
|---|---|---|
| Load | `local_load_*` | `device_load_*` |
| Store | `local_store_*` | `device_store_*` |
| Atomic update | `local_atomic_*` | `atomic_*` |
| Synchronization | Workgroup barriers and fences | Scoped atomic protocols and fences |

Next: [Control flow](/guides/control-flow/).
