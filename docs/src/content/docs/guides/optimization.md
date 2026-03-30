---
title: Optimization
description: Practical techniques for maximizing WAVE kernel performance - occupancy, memory coalescing, wave ops, and more.
---

Getting a kernel to produce correct results is the first step; making it fast is the second. This guide covers the most impactful optimization techniques for WAVE kernels.

## Register Pressure and Occupancy

Occupancy is the number of waves that can run simultaneously on a single compute unit. Higher occupancy helps hide memory latency because the hardware can switch to another wave while one is waiting on a load.

Occupancy is determined by register usage:

```
O = floor(F / (R * W * w))
```

| Symbol | Meaning |
|---|---|
| `O` | Occupancy (waves per compute unit) |
| `F` | Register file size (total 32-bit registers per compute unit) |
| `R` | Registers used per thread (your `.registers` declaration) |
| `W` | Wave width (lanes per wave, e.g., 32 or 64) |
| `w` | Waves to schedule |

**Example:** If a compute unit has 16,384 registers, your kernel uses 16 registers, and waves are 32 lanes wide:

```
O = floor(16384 / (16 * 32 * 1)) = 32 waves
```

If you increase register usage to 32:

```
O = floor(16384 / (32 * 32 * 1)) = 16 waves
```

Doubling register usage halves occupancy.

### Practical Tips

- **Use `.registers` honestly.** Only declare what you need. The compiler and runtime use this number to compute occupancy.
- **Reuse registers aggressively.** Once a value is consumed and no longer needed, reuse that register for something else.
- **Spill to local memory as a last resort.** If you need more than 32 registers worth of live values, store temporaries to local memory and reload them. This costs ~20 cycles per round trip but frees a register.

```asm
; Spill r15 to local memory, reuse r15, then restore
local_store.u32 r15, r20    ; spill r15 to local addr in r20
; ... use r15 for something else ...
local_load.u32 r15, r20     ; restore r15
```

## Memory Coalescing

When threads in a wave access consecutive memory addresses, the hardware combines (coalesces) them into a single wide transaction. This can improve throughput by 8-32x compared to scattered accesses.

**Coalesced access (fast):** Thread `i` accesses address `base + i * 4`:

```asm
mov_sr r0, sr_thread_id_x
shl    r1, r0, 2            ; byte offset = tid * 4
iadd   r2, r10, r1          ; address = base + tid * 4
device_load.u32 r3, r2      ; coalesced - consecutive addresses
```

**Scattered access (slow):** Thread `i` accesses address `base + hash(i) * 4`:

```asm
; Avoid this pattern - scattered loads cannot coalesce
device_load.u32 r3, r_random_addr
```

### Coalescing Rules of Thumb

- Access consecutive 4-byte elements: thread 0 reads `base+0`, thread 1 reads `base+4`, etc.
- Use wider loads (`u128`) when each thread needs 4 consecutive floats - one `u128` load is faster than four `u32` loads.
- Avoid stride-2 or stride-N access patterns. If your algorithm requires strided access, consider tiling through local memory first.

## Using Local Memory as a Scratchpad

When threads need to share data or when an algorithm requires non-coalesced access patterns, stage the data through local memory:

```asm
; --- Phase 1: Coalesced load from device → local ---
mov_sr r0, sr_thread_id_x
shl    r1, r0, 2
iadd   r2, r10, r1
device_load.u32 r3, r2          ; coalesced device load
local_store.u32 r3, r1          ; write to local memory at tid*4

barrier                          ; sync all threads

; --- Phase 2: Non-sequential read from local (fast, bank-parallel) ---
; Now each thread can read any element from local memory
; at ~20 cycle latency instead of ~300 cycle device latency
shl    r4, r5, 2                ; r5 = some computed index
local_load.u32 r6, r4
```

This pattern is the foundation of optimized matrix transpose, reduction, and stencil kernels.

## Wave Operations for Reductions

Wave operations let threads within a wave communicate directly through the register file, without touching local memory at all. For reductions, this is significantly faster.

### Naive Reduction (Shared Memory)

```asm
; Store each thread's value to local memory, then reduce with a tree
mov_sr r0, sr_thread_id_x
shl    r1, r0, 2
local_store.u32 r5, r1          ; write partial result
barrier                          ; sync

; Tree reduction: log2(N) steps, each with a barrier
; ... many instructions, many barriers ...
```

### Optimized Reduction (Wave Ops)

```asm
; Single instruction reduces across the entire wave - no barriers needed
reduce_add r1, r0               ; r1 = sum of r0 across all lanes in the wave
```

Other useful wave operations:

```asm
; Prefix (inclusive) sum across the wave
prefix_sum r1, r0               ; r1[lane] = r0[0] + r0[1] + ... + r0[lane]

; Broadcast lane 0's value to all lanes
broadcast r1, r0, 0             ; r1 = r0 from lane 0

; Shuffle: read another lane's register
shuffle r1, r0, r2              ; r1 = r0 from the lane whose index is in r2

; Cross-lane shift patterns
shuffle_up   r1, r0, 1          ; r1 = r0 from lane (my_lane - 1)
shuffle_down r1, r0, 1          ; r1 = r0 from lane (my_lane + 1)
shuffle_xor  r1, r0, 1          ; r1 = r0 from lane (my_lane XOR 1) - butterfly

; Ballot: gather predicates across the wave
ballot r1, p0                   ; r1 = bitmask of which lanes have p0 == true

; Collective predicates
any p1, p0                      ; p1 = true if any lane has p0 set
all p1, p0                      ; p1 = true if all lanes have p0 set
```

**Guideline:** Always reduce within a wave using `reduce_add` / `reduce_min` / `reduce_max` before falling back to shared-memory tree reduction across waves.

## Minimizing Divergence

Divergent branches force the wave to execute both paths. A few strategies to reduce divergence cost:

### 1. Use `select` for Simple Conditionals

```asm
; Bad: branch diverges the wave
icmp.lt p0, r0, r1
if p0
  mov_imm r2, 1
else
  mov_imm r2, 0
endif

; Good: no divergence
icmp.lt p0, r0, r1
select r2, p0, r3, r4    ; branchless
```

### 2. Reorganize Work to Align Branches with Wave Boundaries

If threads 0–31 always take one path and threads 32–63 always take the other, neither wave diverges. Structure your workload so that nearby thread IDs follow the same control path when possible.

### 3. Avoid Variable-Length Loops

If different threads iterate different numbers of times, the wave runs for as many iterations as the slowest lane. Equalize work per thread, or batch threads with similar iteration counts together.

## Cache Hints

Use cache hints to give the hardware better information about your access pattern:

| Pattern | Recommended hint |
|---|---|
| Data reused multiple times | `cached` (default) |
| Large sequential scan, each element read once | `streaming` |
| Data written by another workgroup | `uncached` |
| Write-only output buffers | `streaming` on stores |

```asm
; Streaming load for a single-pass scan
device_load.u32.streaming r0, r1

; Uncached load to see another workgroup's atomic updates
device_load.u32.uncached r0, r1
```

## Optimization Checklist

Here is a prioritized checklist for optimizing a WAVE kernel:

1. **Coalesce memory accesses.** This is almost always the single largest performance factor. Ensure adjacent threads access adjacent addresses.
2. **Use wave ops for reductions.** Replace shared-memory tree reductions with `reduce_add`, `prefix_sum`, etc.
3. **Minimize register usage.** Lower register count means higher occupancy and better latency hiding.
4. **Tile through local memory.** When you cannot coalesce device loads directly, load a tile cooperatively, barrier, then access from local memory.
5. **Reduce divergence.** Use `select` for simple branches. Align branch conditions with wave boundaries when possible.
6. **Choose appropriate cache hints.** Use `streaming` for single-pass data and `uncached` when freshness matters.
7. **Prefer `fma`/`imad` over separate multiply-add.** Fused multiply-add is a single instruction with better precision (for floats) and throughput.
8. **Use half-precision when possible.** `hadd2` and `hmul2` process two half-precision values per instruction, doubling throughput.

Next: [Python SDK](/guides/python-sdk/) - learn how to launch WAVE kernels from Python with the `wave_gpu` package.
