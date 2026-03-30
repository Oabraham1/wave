---
title: Writing Kernels
description: Learn WAVE assembly syntax, kernel structure, register usage, and how to write your first GPU kernel from scratch.
---

WAVE kernels are short assembly programs that run in parallel across thousands of threads on a GPU. This guide covers the assembly syntax, kernel structure, register conventions, and walks through a complete vector addition kernel line by line.

## Kernel Structure

Every kernel is enclosed in `.kernel` and `.end` directives and declares its resource requirements up front:

```asm
.kernel vector_add
  .registers 8
  .workgroup_size 256

  ; kernel body here

.end
```

- **`.kernel <name>`** begins a named kernel.
- **`.registers <n>`** declares how many general-purpose registers (GPRs) the kernel uses (out of the 32 available, `r0`–`r31`).
- **`.workgroup_size <n>`** sets the number of threads per workgroup (commonly 64, 128, or 256).
- **`.end`** closes the kernel definition.

## Registers

WAVE provides three categories of registers:

### General-Purpose Registers (r0–r31)

All 32 GPRs are untyped 32-bit slots. The instruction determines how the bits are interpreted - `iadd` treats them as integers, `fadd` treats them as IEEE 754 floats, and so on. There is no need to declare types.

```asm
mov_imm r0, 42        ; r0 now holds integer 42
iadd r2, r0, r1       ; integer add: r2 = r0 + r1
fadd r5, r3, r4       ; float add:  r5 = r3 + r4
```

### Predicate Registers (p0–p3)

Four 1-bit predicate registers store comparison results and control conditional execution:

```asm
icmp.lt p0, r0, r1    ; p0 = (r0 < r1)
```

### Special Registers

Sixteen read-only special registers expose the thread's position in the dispatch grid:

| Register | Description |
|---|---|
| `sr_thread_id_x`, `sr_thread_id_y`, `sr_thread_id_z` | Thread index within the workgroup |
| `sr_wave_id` | Wave index within the workgroup |
| `sr_lane_id` | Lane index within the wave |
| `sr_workgroup_id_x`, `sr_workgroup_id_y`, `sr_workgroup_id_z` | Workgroup index within the grid |
| `sr_workgroup_size_x`, `sr_workgroup_size_y`, `sr_workgroup_size_z` | Workgroup dimensions |
| `sr_grid_size_x`, `sr_grid_size_y`, `sr_grid_size_z` | Grid dimensions |
| `sr_wave_width` | Number of lanes per wave (hardware-dependent) |
| `sr_num_waves` | Number of waves in the workgroup |

Access them with `mov_sr`:

```asm
mov_sr r0, sr_thread_id_x       ; r0 = local thread index
mov_sr r1, sr_workgroup_id_x    ; r1 = workgroup index
mov_sr r2, sr_workgroup_size_x  ; r2 = workgroup size
```

## Computing a Global Thread Index

Most kernels need a unique global index per thread. The standard pattern is:

```asm
mov_sr r0, sr_workgroup_id_x     ; workgroup index
mov_sr r1, sr_workgroup_size_x   ; workgroup size
mov_sr r2, sr_thread_id_x        ; local thread index
imul   r3, r0, r1                ; r3 = workgroup_id * workgroup_size
iadd   r3, r3, r2                ; r3 = global thread index
```

Or in one step with `imad` (integer multiply-add):

```asm
mov_sr r0, sr_workgroup_id_x
mov_sr r1, sr_workgroup_size_x
mov_sr r2, sr_thread_id_x
imad   r3, r0, r1, r2            ; r3 = workgroup_id * workgroup_size + thread_id
```

## Loading and Storing Data

WAVE distinguishes two address spaces for memory operations:

- **`device_load` / `device_store`** access global GPU memory (64-bit addresses).
- **`local_load` / `local_store`** access per-workgroup shared memory.

Both support widths `u8`, `u16`, `u32`, `u64`, and `u128`. A cache hint can be appended: `cached` (default), `uncached`, or `streaming`.

```asm
; Load a 32-bit value from device memory at address in r1 into r0
device_load.u32 r0, r1

; Store a 32-bit value from r0 to device memory at address in r1
device_store.u32 r0, r1

; Uncached load (bypass L1 cache)
device_load.u32.uncached r0, r1
```

### Address Arithmetic

Device memory uses 64-bit addresses. To index into an array of 32-bit elements, shift the global index left by 2 (multiply by 4 bytes) and add it to the base pointer:

```asm
; r3 = global thread index, r4 = base address of array
shl  r5, r3, 2          ; r5 = r3 * 4 (byte offset for u32 elements)
iadd r5, r5, r4         ; r5 = &array[global_index]
device_load.u32 r6, r5  ; r6 = array[global_index]
```

## Complete Example: Vector Add

This kernel computes `C[i] = A[i] + B[i]` for each element:

```asm
.kernel vector_add
  .registers 8
  .workgroup_size 256

  ; --- Step 1: Compute global thread index ---
  mov_sr r0, sr_workgroup_id_x       ; r0 = workgroup index
  mov_sr r1, sr_workgroup_size_x     ; r1 = threads per workgroup (256)
  mov_sr r2, sr_thread_id_x          ; r2 = thread index within workgroup
  imad   r3, r0, r1, r2              ; r3 = global_id = workgroup_id * 256 + thread_id

  ; --- Step 2: Compute byte offset for 32-bit floats ---
  shl    r4, r3, 2                   ; r4 = global_id * 4 (byte offset)

  ; --- Step 3: Load A[i] ---
  ; Assume r10 holds base address of A (set by runtime)
  iadd   r5, r10, r4                 ; r5 = &A[global_id]
  device_load.u32 r0, r5            ; r0 = A[global_id]

  ; --- Step 4: Load B[i] ---
  ; Assume r11 holds base address of B (set by runtime)
  iadd   r5, r11, r4                 ; r5 = &B[global_id]
  device_load.u32 r1, r5            ; r1 = B[global_id]

  ; --- Step 5: Add ---
  fadd   r2, r0, r1                  ; r2 = A[i] + B[i]

  ; --- Step 6: Store C[i] ---
  ; Assume r12 holds base address of C (set by runtime)
  iadd   r5, r12, r4                 ; r5 = &C[global_id]
  device_store.u32 r2, r5           ; C[global_id] = r2

  ; --- Done ---
  return

.end
```

**Line-by-line breakdown:**

1. **Lines 6–9**: Read the workgroup ID, workgroup size, and thread ID from special registers, then combine them with `imad` to get a unique global index.
2. **Line 12**: Shift left by 2 converts the element index into a byte offset (each `f32` is 4 bytes).
3. **Lines 15–16**: Add the byte offset to `A`'s base address and load the element.
4. **Lines 19–20**: Same for `B`.
5. **Line 23**: `fadd` performs a 32-bit floating-point addition.
6. **Lines 26–27**: Write the result to `C`.
7. **Line 30**: `return` terminates the thread.

## Instruction Quick Reference

Here are the most common instruction categories you will use in kernels:

| Category | Instructions |
|---|---|
| Integer arithmetic | `iadd`, `isub`, `imul`, `imul_hi`, `imad`, `idiv`, `imod`, `ineg`, `iabs` |
| Float arithmetic | `fadd`, `fsub`, `fmul`, `fma`, `fdiv`, `fneg`, `fabs`, `fsqrt` |
| Float unary | `frsqrt`, `frcp`, `ffloor`, `fceil`, `fround`, `ftrunc`, `ffract`, `fsat`, `fsin`, `fcos`, `fexp2`, `flog2` |
| Bitwise | `and`, `or`, `xor`, `not`, `shl`, `shr`, `sar` |
| Compare | `icmp`, `ucmp`, `fcmp` with conditions: `eq`, `ne`, `lt`, `le`, `gt`, `ge` |
| Data movement | `mov`, `mov_imm`, `mov_sr`, `cvt` |
| Control flow | `if`, `else`, `endif`, `loop`, `break`, `continue`, `endloop` |
| Synchronization | `barrier`, `fence_acquire`, `fence_release`, `return`, `halt` |

## Next Steps

Next: [Memory Model](/guides/memory-model/) - understand the three memory levels, atomic operations, and how fences keep your data consistent.
