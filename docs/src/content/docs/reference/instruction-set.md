---
title: Instruction Set Reference
description: Complete WAVE ISA instruction set organized by category.
---

This is the complete instruction reference for the WAVE ISA v0.3. Each instruction is listed with its mnemonic, opcode, format, operands, and description.

## Instruction Formats

**Base format (32-bit):** Used for register-to-register operations.

| Bits | Field | Width |
|------|-------|-------|
| 31-24 | Opcode | 8 |
| 23-20 | Modifier | 4 |
| 19-18 | Flags | 2 |
| 17-13 | Dst (rd) | 5 |
| 12-8 | Src1 (rs1) | 5 |
| 7-3 | Src2 (rs2) | 5 |
| 2-0 | Reserved | 3 |

**Extended format (64-bit):** Base word followed by a 32-bit immediate. Used for load/store offsets, branch targets, and large constants.

## Operand Notation

| Notation | Meaning |
|----------|---------|
| `rd` | Destination register (`r0`-`r31`) |
| `rs1` | First source register |
| `rs2` | Second source register |
| `imm32` | 32-bit immediate (extended format) |
| `[rs1 + imm32]` | Memory address: base register plus immediate offset |

---

## Integer Arithmetic (0x00 - 0x0B)

| Mnemonic | Opcode | Format | Operands | Description |
|----------|--------|--------|----------|-------------|
| `ADD` | `0x00` | Base | `rd, rs1, rs2` | Integer addition. `rd = rs1 + rs2` |
| `SUB` | `0x01` | Base | `rd, rs1, rs2` | Integer subtraction. `rd = rs1 - rs2` |
| `MUL` | `0x02` | Base | `rd, rs1, rs2` | Integer multiplication (low 32 bits). `rd = (rs1 * rs2) & 0xFFFFFFFF` |
| `DIV` | `0x03` | Base | `rd, rs1, rs2` | Signed integer division. `rd = rs1 / rs2`. Undefined if `rs2 == 0`. |
| `UDIV` | `0x04` | Base | `rd, rs1, rs2` | Unsigned integer division. `rd = rs1 / rs2` (unsigned). |
| `REM` | `0x05` | Base | `rd, rs1, rs2` | Signed integer remainder. `rd = rs1 % rs2`. |
| `UREM` | `0x06` | Base | `rd, rs1, rs2` | Unsigned integer remainder. |
| `NEG` | `0x07` | Base | `rd, rs1` | Integer negation. `rd = -rs1`. `rs2` ignored. |
| `ABS` | `0x08` | Base | `rd, rs1` | Absolute value. `rd = |rs1|` (signed). `rs2` ignored. |
| `MIN` | `0x09` | Base | `rd, rs1, rs2` | Signed minimum. `rd = min(rs1, rs2)`. |
| `MAX` | `0x0A` | Base | `rd, rs1, rs2` | Signed maximum. `rd = max(rs1, rs2)`. |
| `MULHI` | `0x0B` | Base | `rd, rs1, rs2` | Integer multiplication (high 32 bits). `rd = (rs1 * rs2) >> 32` |

---

## Floating-Point (0x10 - 0x1F)

Modifier bits select rounding mode: `0` = round-to-nearest-even (default), `1` = round-toward-zero, `2` = round-toward-positive-infinity, `3` = round-toward-negative-infinity.

| Mnemonic | Opcode | Format | Operands | Description |
|----------|--------|--------|----------|-------------|
| `FADD` | `0x10` | Base | `rd, rs1, rs2` | Floating-point addition. `rd = rs1 + rs2` |
| `FSUB` | `0x11` | Base | `rd, rs1, rs2` | Floating-point subtraction. `rd = rs1 - rs2` |
| `FMUL` | `0x12` | Base | `rd, rs1, rs2` | Floating-point multiplication. `rd = rs1 * rs2` |
| `FDIV` | `0x13` | Base | `rd, rs1, rs2` | Floating-point division. `rd = rs1 / rs2` |
| `FNEG` | `0x14` | Base | `rd, rs1` | Floating-point negation. `rd = -rs1` |
| `FABS` | `0x15` | Base | `rd, rs1` | Floating-point absolute value. `rd = |rs1|` |
| `FSQRT` | `0x16` | Base | `rd, rs1` | Square root. `rd = sqrt(rs1)` |
| `FMIN` | `0x17` | Base | `rd, rs1, rs2` | Floating-point minimum (NaN-propagating). |
| `FMAX` | `0x18` | Base | `rd, rs1, rs2` | Floating-point maximum (NaN-propagating). |
| `FFLOOR` | `0x19` | Base | `rd, rs1` | Floor. `rd = floor(rs1)` |
| `FCEIL` | `0x1A` | Base | `rd, rs1` | Ceiling. `rd = ceil(rs1)` |
| `FROUND` | `0x1B` | Base | `rd, rs1` | Round to nearest integer (ties to even). |
| `FTRUNC` | `0x1C` | Base | `rd, rs1` | Truncate toward zero. |
| `FMA` | `0x1D` | Base | `rd, rs1, rs2` | Fused multiply-add. `rd = rs1 * rs2 + rd`. Note: uses `rd` as the addend (accumulator). |
| `FCVT_I2F` | `0x1E` | Base | `rd, rs1` | Convert signed integer to float. |
| `FCVT_F2I` | `0x1F` | Base | `rd, rs1` | Convert float to signed integer (truncation). |

---

## Bitwise (0x20 - 0x27)

| Mnemonic | Opcode | Format | Operands | Description |
|----------|--------|--------|----------|-------------|
| `AND` | `0x20` | Base | `rd, rs1, rs2` | Bitwise AND. `rd = rs1 & rs2` |
| `OR` | `0x21` | Base | `rd, rs1, rs2` | Bitwise OR. `rd = rs1 \| rs2` |
| `XOR` | `0x22` | Base | `rd, rs1, rs2` | Bitwise XOR. `rd = rs1 ^ rs2` |
| `NOT` | `0x23` | Base | `rd, rs1` | Bitwise NOT. `rd = ~rs1` |
| `SHL` | `0x24` | Base | `rd, rs1, rs2` | Shift left. `rd = rs1 << (rs2 & 31)` |
| `SHR` | `0x25` | Base | `rd, rs1, rs2` | Logical shift right. `rd = rs1 >>> (rs2 & 31)` |
| `SAR` | `0x26` | Base | `rd, rs1, rs2` | Arithmetic shift right. `rd = rs1 >> (rs2 & 31)` (sign-extending) |
| `POPCNT` | `0x27` | Base | `rd, rs1` | Population count. `rd = popcount(rs1)` |

---

## Comparison (0x28 - 0x2C)

Comparison instructions write `1` (true) or `0` (false) to `rd`. Modifier selects comparison predicate for `FCMP`.

| Mnemonic | Opcode | Format | Operands | Description |
|----------|--------|--------|----------|-------------|
| `CMP_EQ` | `0x28` | Base | `rd, rs1, rs2` | Integer equal. `rd = (rs1 == rs2) ? 1 : 0` |
| `CMP_NE` | `0x29` | Base | `rd, rs1, rs2` | Integer not equal. |
| `CMP_LT` | `0x2A` | Base | `rd, rs1, rs2` | Signed less than. |
| `CMP_GE` | `0x2B` | Base | `rd, rs1, rs2` | Signed greater or equal. |
| `FCMP` | `0x2C` | Base | `rd, rs1, rs2` | Floating-point compare. Modifier selects predicate: `0`=EQ, `1`=NE, `2`=LT, `3`=LE, `4`=GT, `5`=GE, `6`=ORD (both not NaN), `7`=UNORD (either NaN). |

---

## Memory (0x30 - 0x31, 0x38 - 0x39)

All memory instructions use the extended format (64-bit) to encode the address offset.

Modifier selects memory ordering: `0` = relaxed, `1` = acquire, `2` = release, `3` = acquire-release, `4` = sequentially consistent.

| Mnemonic | Opcode | Format | Operands | Description |
|----------|--------|--------|----------|-------------|
| `LOAD` | `0x30` | Extended | `rd, [rs1 + imm32]` | Load 32-bit value from global memory into `rd`. |
| `STORE` | `0x31` | Extended | `[rs1 + imm32], rs2` | Store 32-bit value from `rs2` to global memory. `rd` field unused. |
| `LOCAL_LOAD` | `0x38` | Extended | `rd, [rs1 + imm32]` | Load 32-bit value from local (shared) memory. |
| `LOCAL_STORE` | `0x39` | Extended | `[rs1 + imm32], rs2` | Store 32-bit value to local (shared) memory. |

---

## Atomic (0x3C - 0x3D)

Atomic instructions operate on global memory. Modifier selects the atomic operation variant.

| Mnemonic | Opcode | Format | Operands | Description |
|----------|--------|--------|----------|-------------|
| `ATOMIC` | `0x3C` | Extended | `rd, [rs1 + imm32], rs2` | Atomic read-modify-write. Loads the value at `[rs1 + imm32]`, writes the old value to `rd`, and applies the operation with `rs2`. Modifier selects operation: `0`=ADD, `1`=SUB, `2`=AND, `3`=OR, `4`=XOR, `5`=MIN, `6`=MAX, `7`=UMIN, `8`=UMAX, `9`=XCHG. |
| `ATOMIC_CAS` | `0x3D` | Extended | `rd, [rs1 + imm32], rs2` | Atomic compare-and-swap. Compares value at `[rs1 + imm32]` with `rd`; if equal, stores `rs2`. Old value written to `rd` in either case. |

---

## Wave Operations (0x3E)

Wave (subgroup) operations execute across all active lanes in a wave. The modifier selects the wave operation variant. `rs2` is unused for broadcast; it serves as the shuffle source lane for `SHUFFLE`.

| Mnemonic | Opcode | Format | Operands | Description |
|----------|--------|--------|----------|-------------|
| `WAVE_OP` | `0x3E` | Base | `rd, rs1, rs2` | Wave-level operation. Modifier selects variant: `0`=BROADCAST (broadcast `rs1` from lane 0 to all lanes), `1`=REDUCE_ADD (sum `rs1` across all active lanes), `2`=REDUCE_MIN, `3`=REDUCE_MAX, `4`=PREFIX_SUM (exclusive prefix sum of `rs1`), `5`=SHUFFLE (read `rs1` from lane `rs2`), `6`=BALLOT (set bit `i` if lane `i` has `rs1 != 0`), `7`=ANY (1 if any lane has `rs1 != 0`), `8`=ALL (1 if all lanes have `rs1 != 0`). |

---

## Control Flow and Synchronization (0x3F)

The modifier on opcode `0x3F` selects the specific control flow or synchronization operation.

| Mnemonic | Opcode | Format | Modifier | Operands | Description |
|----------|--------|--------|----------|----------|-------------|
| `BARRIER` | `0x3F` | Base | `0` | *(none)* | Workgroup barrier. All threads in the workgroup must reach this point before any proceed. |
| `BRANCH` | `0x3F` | Extended | `1` | `imm32` | Unconditional branch. Sets PC to `imm32`. |
| `BRANCH_IF` | `0x3F` | Extended | `2` | `rs1, imm32` | Conditional branch. If `rs1 != 0`, sets PC to `imm32`. |
| `BRANCH_IFNOT` | `0x3F` | Extended | `3` | `rs1, imm32` | Conditional branch. If `rs1 == 0`, sets PC to `imm32`. |
| `CALL` | `0x3F` | Extended | `4` | `imm32` | Push return address and branch to `imm32`. |
| `RET` | `0x3F` | Base | `5` | *(none)* | Pop return address and branch to it. |
| `EXIT` | `0x3F` | Base | `6` | *(none)* | Terminate the current thread. |
| `NOP` | `0x3F` | Base | `7` | *(none)* | No operation. |

---

## Opcode Map Summary

| Range | Category | Count |
|-------|----------|-------|
| `0x00`-`0x0B` | Integer Arithmetic | 12 |
| `0x10`-`0x1F` | Floating-Point | 16 |
| `0x20`-`0x27` | Bitwise | 8 |
| `0x28`-`0x2C` | Comparison | 5 |
| `0x30`-`0x31` | Global Memory | 2 |
| `0x38`-`0x39` | Local Memory | 2 |
| `0x3C`-`0x3D` | Atomic | 2 |
| `0x3E` | Wave Operations | 1 (9 variants) |
| `0x3F` | Control Flow / Sync | 1 (8 variants) |
| | **Total** | **49 opcodes** |
