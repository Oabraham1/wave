---
title: "Optimization"
description: "Register usage, memory access, wave operations, and divergence."
---

Measure a representative workload, identify its bottleneck, and check that each optimization preserves the kernel's numerical and memory behavior.

## Register pressure and occupancy

Register allocation limits how many waves can reside on a compute unit. More resident waves can help hide memory latency, but other limits also apply, including local memory, workgroup size, and hardware scheduling capacity.

Ignoring allocation granularity, the register limit is:

```text
resident_waves <= floor(registers_per_unit / (registers_per_thread * wave_width))
```

For 16,384 registers, 16 registers per thread, and 32 lanes per wave, this bound is 32 waves. Doubling the per-thread allocation reduces the bound to 16. Use the vendor compiler's resource report to find the actual allocation; the WAVE `.registers` declaration alone does not determine hardware occupancy.

### Practical tips

- End value lifetimes early and reuse registers when their previous values are no longer needed.
- Declare enough registers for every operand.
- Measure the tradeoff before moving temporary values into local memory. Extra memory operations and workgroup storage can offset register savings.

A local-memory temporary needs an address unique to each thread:

```asm
local_store_u32 r20, r15
; Reuse r15 here. r20 must remain unchanged.
local_load_u32 r15, r20
```

## Memory coalescing

Adjacent lanes accessing adjacent addresses can share memory transactions. Alignment, element size, and the target determine how many transactions are needed.

```asm
mov_sr r0, sr_thread_id_x
mov_imm r1, 4
imul r2, r0, r1
iadd r3, r10, r2
device_load_u32 r4, r3
```

Here `r10` is the array's base offset and each thread reads one consecutive word.

### Coalescing rules of thumb

- Map adjacent lanes to adjacent elements where the algorithm permits it.
- Consider wider loads when each thread needs adjacent words and the address is aligned.
- For strided or transposed access, test a tile staged through local memory.

## Using local memory as a scratchpad

A cooperative tile separates device access from the order in which threads consume the data:

```asm
mov_sr r0, sr_thread_id_x
mov_imm r1, 4
imul r2, r0, r1
iadd r3, r10, r2
device_load_u32 r4, r3
local_store_u32 r2, r4
barrier

; r5 is a valid element index within the shared tile.
imul r6, r5, r1
local_load_u32 r7, r6
```

Declare enough local memory for the tile. Add another barrier before reusing the storage if other threads may still be reading it.

## Wave operations for reductions

Wave operations exchange values between lanes without a shared-memory array. For a workgroup reduction, first combine values within each wave, then combine the wave results through local memory.

### Naive reduction: shared memory

A shared-memory tree stores partial values and synchronizes at each stage. It is useful across waves, where register exchange is unavailable. Store and read phases must be separated by barriers, including before shared storage is reused.

### Optimized reduction: wave ops

For a known wave width of 32, this XOR tree sums `f32` values across a fully active wave:

```asm
mov_imm r1, 16
wave_shuffle_xor r2, r0, r1
fadd r0, r0, r2
mov_imm r1, 8
wave_shuffle_xor r2, r0, r1
fadd r0, r0, r2
mov_imm r1, 4
wave_shuffle_xor r2, r0, r1
fadd r0, r0, r2
mov_imm r1, 2
wave_shuffle_xor r2, r0, r1
fadd r0, r0, r2
mov_imm r1, 1
wave_shuffle_xor r2, r0, r1
fadd r0, r0, r2
```

Adapt the tree to the target wave width and initialize inactive data lanes to zero before the collective. Changing the reduction order can change floating-point results.

WAVE also provides `wave_reduce_add`, `wave_reduce_min`, `wave_reduce_max`, `wave_prefix_sum`, `wave_broadcast`, and predicate collectives. Check the instruction's operand types and active-lane requirements before substituting it.

## Minimizing divergence

Divergence leaves some lanes inactive while others execute a path.

### 1. Use `select` for simple conditionals

```asm
mov_imm r3, 1
mov_imm r4, 0
icmp_lt p0, r0, r1
select r2, p0, r3, r4
```

`select` is useful when both candidate values can be computed safely. Measure it against branching when producing those values is expensive.

### 2. Reorganize work to align branches with wave boundaries

Group inputs with similar control flow when the reordering cost is justified. A branch uniform within each wave does not split that wave's active lanes.

### 3. Avoid variable-length loops

A wave with different per-lane iteration counts remains occupied until its last participating lane finishes. Group similar workloads or redistribute work when practical.

## Cache hints

Use `cached` for reuse and consider `streaming` for large sequential accesses. `uncached` requests cache bypass. Hints affect cache policy; synchronization still requires the appropriate atomic operations, fences, and barriers.

```asm
device_load_u32.streaming r0, r1
```

## Optimization checklist

1. Measure representative sizes and input distributions.
2. Reduce unnecessary transfers and improve device-memory access patterns.
3. Use wave communication where its numerical and lane semantics match the algorithm.
4. Check register allocation, local-memory usage, and workgroup size together.
5. Reduce divergence where it limits useful lane activity.
6. Compare timing distributions across repeated runs.
7. Use `fma` only when fused rounding matches the algorithm. Replacing separate multiply and add changes floating-point semantics.
8. Use lower precision only when the accuracy requirements permit it.

Next: [Python SDK](/guides/python-sdk/).
