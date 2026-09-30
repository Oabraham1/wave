---
title: "Register model"
description: "WAVE registers, register pairs, and allocation."
---

WAVE general-purpose registers hold untyped 32-bit values. The instruction determines how it reads those bits. An integer add reads integers; a floating-point add reads IEEE 754 values.

## Register addressing

The 8-bit register fields address `r0` through `r255`. The implementation's `MAX_REGISTERS` capability determines the available count. Kernel metadata declares the registers the kernel needs. See [Binary encoding](/internals/binary-encoding/).

`r0` is an ordinary writable register. An atomic instruction with destination zero discards its return value.

## Wide values

A 64-bit value occupies consecutive registers. The low register holds the low 32 bits. The instruction defines how it interprets the pair.

## Allocation and occupancy

The compiler assigns values to WAVE register indices within the configured limit. It returns an error when allocation fails; it does not insert spills. The vendor compiler then allocates machine registers, which affect occupancy. See [Occupancy equation](/internals/occupancy-equation/).

Predicate registers and special registers have separate roles. Predicates control lane execution; special registers expose thread identifiers and device capabilities. See the [specification](/reference/spec-v04/) for their access rules.
