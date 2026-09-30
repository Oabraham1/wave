---
title: "Modifier field"
description: "Operation variants in the WAVE instruction encoding."
---

Word0 bits [7:4] hold a 4-bit modifier. Its meaning depends on the opcode. Values range from 0 through 15.

## Floating-point unary operations

Opcode `0x1B` uses these modifiers:

| Modifier | Operation |
|----------|-----------|
| 0 | `frsqrt` |
| 1 | `frcp` |
| 2 | `ffloor` |
| 3 | `fceil` |
| 4 | `fround` |
| 5 | `ftrunc` |
| 6 | `ffract` |
| 7 | `fsat` |
| 8 | `fsin` |
| 9 | `fcos` |
| 10 | `fexp2` |
| 11 | `flog2` |

`fsqrt`, `fabs`, and `fneg` have separate opcodes.

## Other operation classes

Atomic modifiers select addition, subtraction, minimum, maximum, bitwise AND, OR, XOR, exchange, and compare-and-swap. Compare-and-swap uses value 8.

Wave reductions use modifiers 8 through 13 for addition, minimum, maximum, bitwise AND, OR, and XOR. Control-flow operations use opcode `0x3F` with modifiers 0 through 7. Synchronization operations share that opcode with modifiers 8 through 15.

The modifier does not overlap register or predicate fields. Memory scope belongs to word1 for scoped instructions. See [Binary encoding](/internals/binary-encoding/) for the complete layout.
