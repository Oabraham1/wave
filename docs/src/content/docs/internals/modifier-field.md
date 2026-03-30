---
title: Modifier Field Evolution
description: How a 3-bit modifier field caused encoding failures and why v0.3 expanded it to 4 bits.
---

The expansion of the modifier field from 3 bits to 4 bits in WAVE v0.3 was driven by a concrete encoding failure: the `FUnaryOp` opcode class has 12 variants, and 3 bits can only represent 8.

## The Problem (v0.2)

In WAVE v0.2, the instruction encoding allocated bits as follows:

```
v0.2 layout:
 31  26 25  21 20  16 15  11 10  8 7  5 4  3  2  1  0
┌──────┬──────┬──────┬──────┬─────┬────┬────┬───┬─────┐
│opcode│  RD  │ RS1  │ RS2  │ mod │ ?? │pred│neg│flags│
│ 6b   │ 5b   │ 5b   │ 5b   │ 3b  │ 3b │ 2b │1b │ 3b │
└──────┴──────┴──────┴──────┴─────┴────┴────┴───┴─────┘
```

The 3-bit modifier field (bits [10:8]) could encode values 0--7. The 3-bit flags field (bits [2:0]) carried instruction-specific flags.

### FUnaryOp variants

The `FUnaryOp` opcode class defines the following floating-point unary operations:

| Modifier | Operation |
|----------|-----------|
| 0 | `fsqrt` |
| 1 | `frsqrt` |
| 2 | `frcp` |
| 3 | `fabs` |
| 4 | `fneg` |
| 5 | `ffloor` |
| 6 | `fceil` |
| 7 | `ftrunc` |
| 8 | `fsin` |
| 9 | `fcos` |
| 10 | `fexp2` |
| 11 | `flog2` |

Variants 0--7 fit in 3 bits. Variants 8--11 (`fsin`, `fcos`, `fexp2`, `flog2`) do not. These operations could not be encoded in v0.2, making the ISA incomplete for basic transcendental math.

### Atomic CAS

The same problem affected atomic operations. Atomic compare-and-swap (`atom.cas`) required modifier value 8 to distinguish it from other atomic variants (add, sub, min, max, and, or, xor, xchg at modifiers 0--7). Under the 3-bit scheme, CAS was unencodable.

## The Fix (v0.3)

Version 0.3 reallocated bits within the lower portion of the instruction word:

```
v0.3 layout:
 31  26 25  21 20  16 15  11 10   7 6  5 4  3  2  1  0
┌──────┬──────┬──────┬──────┬──────┬────┬────┬───┬────┐
│opcode│  RD  │ RS1  │ RS2  │ mod  │scop│pred│neg│flag│
│ 6b   │ 5b   │ 5b   │ 5b   │ 4b   │ 2b │ 2b │1b │ 2b│
└──────┴──────┴──────┴──────┴──────┴────┴────┴───┴────┘
```

The changes:

1. **Modifier expanded from 3 to 4 bits** (bits [10:7]). This encodes values 0--15, covering all 12 `FUnaryOp` variants and all atomic variants including CAS with room to spare.

2. **Flags reduced from 3 to 2 bits** (bits [1:0]). Two flags were eliminated:
   - `WAVE_REDUCE_FLAG`: originally indicated a reduction operation. This was folded into the opcode/modifier scheme instead, since reductions are distinct operations, not flags on existing operations.
   - `NON_RETURNING_ATOMIC_FLAG`: originally indicated an atomic that discards its result. This was made implicit --- if `RD` is the zero register, the result is discarded.

3. **Scope field formalized** (bits [6:5]). The previously ambiguous 3 bits between modifier and predicate were split into a 2-bit explicit scope field, giving memory scoping first-class representation in the encoding.

## Why Not a Wider Instruction Word?

Expanding to 48-bit or variable-width instructions was considered and rejected:

- **Decode complexity**: Fixed-width 32-bit instructions allow the decoder to process one instruction per cycle per lane with no alignment logic. Variable-width decoding requires a prefix scan to find instruction boundaries.
- **Instruction cache efficiency**: 32-bit instructions maximize I-cache utilization. Wider instructions reduce the number of instructions per cache line.
- **Vendor precedent**: All four target architectures use fixed-width instruction words (32-bit for NVIDIA and AMD scalar, 128-bit for Intel and AMD vector, 32-bit for Apple). The WAVE encoding aligns with the most common width.

The 64-bit extended format (see [Binary Encoding](/internals/binary-encoding/)) handles the cases that genuinely need more bits, without penalizing the common case.

## Lessons Learned

This encoding defect illustrates a general principle in ISA design: bit allocation must be validated against the full enumeration of every opcode class's variants, not just the common ones. The `fsin`/`fcos`/`fexp2`/`flog2` operations are less common than `fadd` or `fmul`, but they are not optional --- any scientific or graphics workload requires transcendental functions. The fix was cheap (reallocating bits within the same 32-bit word) because it was caught before hardware implementation.
