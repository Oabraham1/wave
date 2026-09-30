---
title: "Encoding and execution invariants"
description: "The rules shared by the assembler, decoder, emulator, and backends."
---

The toolchain uses the [v0.4 specification](/reference/spec-v04/). These invariants apply across assembly, decoding, and execution.

## Register fields

Each register field is 8 bits and addresses `r0` through `r255`. Kernel metadata declares the register allocation. General-purpose register `r0` is writable. Non-returning atomic instructions use a destination field of zero to discard the result.

## Modifiers and predicates

Word0 bits [7:4] select an operation variant. Bits [1:0] select a predicate register, bit [2] negates it, and bit [3] is reserved. A zero selector with no negation means unpredicated execution. Predicate register `p0` is writable and can be a comparison destination or control-flow operand.

The control opcode uses modifiers 0 through 7 for control flow and 8 through 15 for synchronization. Miscellaneous operations use opcode `0x41`. Scoped instructions encode scope in word1 bits [1:0]. See [Modifier field](/internals/modifier-field/).

## Wave state

Each wave owns its program counter, active mask, and control flow stacks. One wave's branches cannot alter another wave's state. Workgroup barriers coordinate the waves in that workgroup.

## Conversion direction

`cvt_A_B` converts type B to type A. For example, `cvt_f32_i32` reads a signed integer and writes its floating-point representation. `cvt_i32_f32` performs the reverse conversion.

## Verification

The specification tests cover field boundaries, register access, predicates, independent waves, barriers, and conversions. Assembly round trips check that decoding and reassembly preserve instruction bytes.
