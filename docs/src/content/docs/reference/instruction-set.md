---
title: "Instruction set reference"
description: "WAVE assembler mnemonics, opcodes, and operands."
---

The assembler accepts the mnemonics below. The [v0.4 specification](/reference/spec-v04/) defines their execution semantics. Instructions contain one or two 32-bit words; see [Binary encoding](/internals/binary-encoding/) for field layouts.

## Operands

| Name | Meaning |
|------|---------|
| `rd` | Destination general-purpose register |
| `rs1` through `rs4` | Source registers or addresses, as defined by the instruction |
| `pd` | Destination predicate register |
| `ps` | Source predicate register |
| `imm32` | 32-bit immediate |
| `label` | Label resolved by the assembler |
| `scope` | Memory scope |
| `optional_rd` | Optional atomic result register |
| `special_register` | Named special register |

General-purpose registers range from `r0` through `r255`. Predicate registers range from `p0` through `p3`. Prefix an instruction with `@p1` or `@!p1` to predicate it. A zero predicate selector with no negation means unpredicated execution; it does not make the writable register `p0` constant.

Memory suffixes select access width. Conversion names put the destination type first: `cvt_f32_i32` converts an integer to a float.

## Integer arithmetic

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `iadd` | `0x00` | `rd, rs1, rs2` |
| `isub` | `0x01` | `rd, rs1, rs2` |
| `imul` | `0x02` | `rd, rs1, rs2` |
| `imul_hi` | `0x03` | `rd, rs1, rs2` |
| `imad` | `0x04` | `rd, rs1, rs2, rs3` |
| `idiv` | `0x05` | `rd, rs1, rs2` |
| `imod` | `0x06` | `rd, rs1, rs2` |
| `ineg` | `0x07` | `rd, rs1` |
| `iabs` | `0x08` | `rd, rs1` |
| `imin` | `0x09` | `rd, rs1, rs2` |
| `imax` | `0x0A` | `rd, rs1, rs2` |
| `iclamp` | `0x0B` | `rd, rs1, rs2, rs3` |

## Floating-point arithmetic

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `fadd` | `0x10` | `rd, rs1, rs2` |
| `fsub` | `0x11` | `rd, rs1, rs2` |
| `fmul` | `0x12` | `rd, rs1, rs2` |
| `fma` | `0x13` | `rd, rs1, rs2, rs3` |
| `fdiv` | `0x14` | `rd, rs1, rs2` |
| `fneg` | `0x15` | `rd, rs1` |
| `fabs` | `0x16` | `rd, rs1` |
| `fmin` | `0x17` | `rd, rs1, rs2` |
| `fmax` | `0x18` | `rd, rs1, rs2` |
| `fclamp` | `0x19` | `rd, rs1, rs2, rs3` |
| `fsqrt` | `0x1A` | `rd, rs1` |
| `fceil` | `0x1B` | `rd, rs1` |
| `fcos` | `0x1B` | `rd, rs1` |
| `fexp2` | `0x1B` | `rd, rs1` |
| `ffloor` | `0x1B` | `rd, rs1` |
| `ffract` | `0x1B` | `rd, rs1` |
| `flog2` | `0x1B` | `rd, rs1` |
| `frcp` | `0x1B` | `rd, rs1` |
| `fround` | `0x1B` | `rd, rs1` |
| `frsqrt` | `0x1B` | `rd, rs1` |
| `fsat` | `0x1B` | `rd, rs1` |
| `fsin` | `0x1B` | `rd, rs1` |
| `ftrunc` | `0x1B` | `rd, rs1` |
| `hadd` | `0x1C` | `rd, rs1, rs2` |
| `hma` | `0x1C` | `rd, rs1, rs2, rs3` |
| `hmul` | `0x1C` | `rd, rs1, rs2` |
| `hsub` | `0x1C` | `rd, rs1, rs2` |
| `hadd2` | `0x1D` | `rd, rs1, rs2` |
| `hma2` | `0x1D` | `rd, rs1, rs2, rs3` |
| `hmul2` | `0x1D` | `rd, rs1, rs2` |
| `dadd` | `0x1E` | `rd, rs1, rs2` |
| `dma` | `0x1E` | `rd, rs1, rs2, rs3` |
| `dmul` | `0x1E` | `rd, rs1, rs2` |
| `dsub` | `0x1E` | `rd, rs1, rs2` |
| `ddiv` | `0x1F` | `rd, rs1, rs2` |
| `dsqrt` | `0x1F` | `rd, rs1` |
| `badd` | `0x2D` | `rd, rs1, rs2` |
| `bma` | `0x2D` | `rd, rs1, rs2, rs3` |
| `bmul` | `0x2D` | `rd, rs1, rs2` |
| `bsub` | `0x2D` | `rd, rs1, rs2` |
| `badd2` | `0x2E` | `rd, rs1, rs2` |
| `bma2` | `0x2E` | `rd, rs1, rs2, rs3` |
| `bmul2` | `0x2E` | `rd, rs1, rs2` |

## Bitwise operations

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `and` | `0x20` | `rd, rs1, rs2` |
| `or` | `0x21` | `rd, rs1, rs2` |
| `xor` | `0x22` | `rd, rs1, rs2` |
| `not` | `0x23` | `rd, rs1` |
| `shl` | `0x24` | `rd, rs1, rs2` |
| `shr` | `0x25` | `rd, rs1, rs2` |
| `sar` | `0x26` | `rd, rs1, rs2` |
| `bfe` | `0x27` | `rd, rs1, rs2, rs3` |
| `bfi` | `0x27` | `rd, rs1, rs2, rs3, rs4` |
| `bitcount` | `0x27` | `rd, rs1` |
| `bitfind` | `0x27` | `rd, rs1` |
| `bitrev` | `0x27` | `rd, rs1` |

## Comparisons and selection

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `icmp_eq` | `0x28` | `pd, rs1, rs2` |
| `icmp_ge` | `0x28` | `pd, rs1, rs2` |
| `icmp_gt` | `0x28` | `pd, rs1, rs2` |
| `icmp_le` | `0x28` | `pd, rs1, rs2` |
| `icmp_lt` | `0x28` | `pd, rs1, rs2` |
| `icmp_ne` | `0x28` | `pd, rs1, rs2` |
| `ucmp_le` | `0x29` | `pd, rs1, rs2` |
| `ucmp_lt` | `0x29` | `pd, rs1, rs2` |
| `fcmp_eq` | `0x2A` | `pd, rs1, rs2` |
| `fcmp_gt` | `0x2A` | `pd, rs1, rs2` |
| `fcmp_le` | `0x2A` | `pd, rs1, rs2` |
| `fcmp_lt` | `0x2A` | `pd, rs1, rs2` |
| `fcmp_ne` | `0x2A` | `pd, rs1, rs2` |
| `fcmp_ord` | `0x2A` | `pd, rs1, rs2` |
| `fcmp_unord` | `0x2A` | `pd, rs1, rs2` |
| `select` | `0x2B` | `rd, ps, rs1, rs2` |

## Conversions

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `cvt_bf16_f32` | `0x2C` | `rd, rs1` |
| `cvt_f16_f32` | `0x2C` | `rd, rs1` |
| `cvt_f32_bf16` | `0x2C` | `rd, rs1` |
| `cvt_f32_f16` | `0x2C` | `rd, rs1` |
| `cvt_f32_f64` | `0x2C` | `rd, rs1` |
| `cvt_f32_i32` | `0x2C` | `rd, rs1` |
| `cvt_f32_u32` | `0x2C` | `rd, rs1` |
| `cvt_f64_f32` | `0x2C` | `rd, rs1` |
| `cvt_i32_f32` | `0x2C` | `rd, rs1` |
| `cvt_u32_f32` | `0x2C` | `rd, rs1` |

## Memory and atomics

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `local_load_u16` | `0x30` | `rd, rs1` |
| `local_load_u32` | `0x30` | `rd, rs1` |
| `local_load_u64` | `0x30` | `rd, rs1` |
| `local_load_u8` | `0x30` | `rd, rs1` |
| `local_store_u16` | `0x31` | `rs1, rs2` |
| `local_store_u32` | `0x31` | `rs1, rs2` |
| `local_store_u64` | `0x31` | `rs1, rs2` |
| `local_store_u8` | `0x31` | `rs1, rs2` |
| `device_load_u128` | `0x38` | `rd, rs1` |
| `device_load_u16` | `0x38` | `rd, rs1` |
| `device_load_u32` | `0x38` | `rd, rs1` |
| `device_load_u64` | `0x38` | `rd, rs1` |
| `device_load_u8` | `0x38` | `rd, rs1` |
| `device_store_u128` | `0x39` | `rs1, rs2` |
| `device_store_u16` | `0x39` | `rs1, rs2` |
| `device_store_u32` | `0x39` | `rs1, rs2` |
| `device_store_u64` | `0x39` | `rs1, rs2` |
| `device_store_u8` | `0x39` | `rs1, rs2` |
| `local_atomic_add` | `0x3C` | `optional_rd, rs1, rs2` |
| `local_atomic_and` | `0x3C` | `optional_rd, rs1, rs2` |
| `local_atomic_cas` | `0x3C` | `optional_rd, rs1, rs2, rs3` |
| `local_atomic_exchange` | `0x3C` | `optional_rd, rs1, rs2` |
| `local_atomic_max` | `0x3C` | `optional_rd, rs1, rs2` |
| `local_atomic_min` | `0x3C` | `optional_rd, rs1, rs2` |
| `local_atomic_or` | `0x3C` | `optional_rd, rs1, rs2` |
| `local_atomic_sub` | `0x3C` | `optional_rd, rs1, rs2` |
| `local_atomic_xor` | `0x3C` | `optional_rd, rs1, rs2` |
| `atomic_add` | `0x3D` | `optional_rd, rs1, rs2, scope` |
| `atomic_and` | `0x3D` | `optional_rd, rs1, rs2, scope` |
| `atomic_cas` | `0x3D` | `optional_rd, rs1, rs2, rs3, scope` |
| `atomic_exchange` | `0x3D` | `optional_rd, rs1, rs2, scope` |
| `atomic_max` | `0x3D` | `optional_rd, rs1, rs2, scope` |
| `atomic_min` | `0x3D` | `optional_rd, rs1, rs2, scope` |
| `atomic_or` | `0x3D` | `optional_rd, rs1, rs2, scope` |
| `atomic_sub` | `0x3D` | `optional_rd, rs1, rs2, scope` |
| `atomic_xor` | `0x3D` | `optional_rd, rs1, rs2, scope` |

## Wave operations

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `wave_all` | `0x3E` | `pd, ps` |
| `wave_any` | `0x3E` | `pd, ps` |
| `wave_ballot` | `0x3E` | `rd, ps` |
| `wave_broadcast` | `0x3E` | `rd, rs1, rs2` |
| `wave_prefix_sum` | `0x3E` | `rd, rs1` |
| `wave_reduce_add` | `0x3E` | `rd, rs1` |
| `wave_reduce_max` | `0x3E` | `rd, rs1` |
| `wave_reduce_min` | `0x3E` | `rd, rs1` |
| `wave_shuffle` | `0x3E` | `rd, rs1, rs2` |
| `wave_shuffle_down` | `0x3E` | `rd, rs1, rs2` |
| `wave_shuffle_up` | `0x3E` | `rd, rs1, rs2` |
| `wave_shuffle_xor` | `0x3E` | `rd, rs1, rs2` |

## Control and synchronization

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `barrier` | `0x3F` | None |
| `break` | `0x3F` | `ps` |
| `call` | `0x3F` | `label` |
| `continue` | `0x3F` | `ps` |
| `else` | `0x3F` | None |
| `endif` | `0x3F` | None |
| `endloop` | `0x3F` | None |
| `fence_acq_rel` | `0x3F` | `scope` |
| `fence_acquire` | `0x3F` | `scope` |
| `fence_release` | `0x3F` | `scope` |
| `halt` | `0x3F` | None |
| `if` | `0x3F` | `ps` |
| `loop` | `0x3F` | None |
| `nop` | `0x3F` | None |
| `return` | `0x3F` | None |
| `wait` | `0x3F` | None |

## Matrix operations

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `mma_compute` | `0x40` | `rd, rs1, rs2` |
| `mma_load_a` | `0x40` | `rd, rs1, rs2` |
| `mma_load_b` | `0x40` | `rd, rs1, rs2` |
| `mma_store_c` | `0x40` | `rd, rs1, rs2` |

## Register moves

| Mnemonic | Opcode | Operands |
|----------|--------|----------|
| `mov` | `0x41` | `rd, rs1` |
| `mov_imm` | `0x41` | `rd, imm32` |
| `mov_sr` | `0x41` | `rd, special_register` |
