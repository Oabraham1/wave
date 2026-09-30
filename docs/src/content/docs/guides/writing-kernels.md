---
title: "Writing kernels"
description: "Assembly syntax, registers, memory operations, and a vector-add kernel."
---

WAVE assembly describes the work performed by each thread. A launch supplies the workgroup grid, memory allocation, and execution width.

## Kernel structure

```asm
.kernel empty
.registers 1
.workgroup_size 256, 1, 1
halt
.end
```

- `.kernel <name>` starts a kernel.
- `.registers <n>` declares the register count.
- `.workgroup_size <x>, <y>, <z>` declares threads per workgroup.
- `.local_memory <bytes>` reserves workgroup storage when needed.
- `.end` closes the kernel definition.

## Registers

### General-purpose registers: r0-r31

The base encoding addresses `r0` through `r31`. Each register holds 32 untyped bits. The instruction selects the interpretation. Extended register encoding supports larger register indices; consult the [specification](/reference/spec/) for its limits.

```asm
mov_imm r0, 42
iadd r2, r0, r1
fadd r5, r3, r4
```

Arithmetic instructions take register operands. Load constants with `mov_imm` before using them in arithmetic.

### Predicate registers: p0-p3

Comparisons write one-bit results to predicate registers:

```asm
icmp_lt p0, r0, r1
```

### Special registers

Special registers describe the current invocation and launch:

| Register | Meaning |
|---|---|
| `sr_thread_id_x`, `sr_thread_id_y`, `sr_thread_id_z` | Thread index within the workgroup |
| `sr_wave_id` | Wave index within the workgroup |
| `sr_lane_id` | Lane index within the wave |
| `sr_workgroup_id_x`, `sr_workgroup_id_y`, `sr_workgroup_id_z` | Workgroup index within the grid |
| `sr_workgroup_size_x`, `sr_workgroup_size_y`, `sr_workgroup_size_z` | Workgroup dimensions |
| `sr_grid_size_x`, `sr_grid_size_y`, `sr_grid_size_z` | Workgroup counts |
| `sr_wave_width` | Lanes per wave |
| `sr_num_waves` | Waves per workgroup |

Read them with `mov_sr`:

```asm
mov_sr r0, sr_thread_id_x
mov_sr r1, sr_workgroup_id_x
mov_sr r2, sr_workgroup_size_x
```

## Computing a global thread index

For a one-dimensional launch, multiply the workgroup index by the workgroup size and add the local thread index:

```asm
mov_sr r0, sr_workgroup_id_x
mov_sr r1, sr_workgroup_size_x
mov_sr r2, sr_thread_id_x
imad r3, r0, r1, r2
```

## Loading and storing data

Device memory holds the launch allocation. Local memory is shared within a workgroup. Loads name the destination first; stores name the address first.

```asm
device_load_u32 r0, r1
device_store_u32 r1, r0
local_load_u32 r2, r3
local_store_u32 r3, r2
```

The width is part of the mnemonic. Device operations support `u8`, `u16`, `u32`, `u64`, and `u128`; local operations support widths through `u64`. Device cache hints follow the mnemonic:

```asm
device_load_u32.uncached r0, r1
```

### Address arithmetic

A 32-bit element occupies four bytes. This example adds a byte offset to the base offset in `r4`:

```asm
mov_imm r7, 4
imul r5, r3, r7
iadd r5, r5, r4
device_load_u32 r6, r5
```

The allocation and argument layout determine base offsets. Do not assume the runtime initializes arbitrary registers to buffer addresses.

## Complete example: vector add

This kernel adds two arrays of 256 `f32` values. Launch one workgroup of 256 threads with a device allocation of at least 3,072 bytes:

| Byte offset | Contents |
|---|---|
| 0 | Input A, 256 floats |
| 1,024 | Input B, 256 floats |
| 2,048 | Output C, 256 floats |

```asm
.kernel vector_add
.registers 16
.workgroup_size 256, 1, 1

mov_sr r0, sr_thread_id_x
mov_imm r1, 4
imul r2, r0, r1
device_load_u32 r3, r2

mov_imm r4, 1024
iadd r5, r4, r2
device_load_u32 r6, r5
fadd r7, r3, r6

mov_imm r8, 2048
iadd r9, r8, r2
device_store_u32 r9, r7
halt
.end
```

Each thread computes its byte offset, loads one value from each input, adds them, and stores one output. For other sizes, pass the element count through a defined argument layout and guard the accesses.

## Instruction quick reference

| Category | Examples |
|---|---|
| Integer arithmetic | `iadd`, `isub`, `imul`, `imad`, `idiv`, `imod` |
| Floating-point arithmetic | `fadd`, `fsub`, `fmul`, `fma`, `fdiv`, `fsqrt` |
| Bit operations | `and`, `or`, `xor`, `not`, `shl`, `shr`, `sar` |
| Comparisons | `icmp_lt`, `ucmp_le`, `fcmp_eq` |
| Data movement | `mov`, `mov_imm`, `mov_sr` |
| Control flow | `if`, `else`, `endif`, `loop`, `break`, `continue`, `endloop` |
| Synchronization and exit | `barrier`, `fence_acquire`, `fence_release`, `return`, `halt` |

## Next steps

Read the [memory model](/guides/memory-model/) for shared storage and synchronization.
