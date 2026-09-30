---
title: "Control flow"
description: "Active lanes, structured branches, and independent wave state."
---

WAVE uses structured branches and loops. Each wave has its own program counter and active mask. Inactive lanes produce no architectural side effects.

## Branches

```asm
icmp_lt p1, r0, r1
if p1
    iadd r2, r2, r3
else
    isub r2, r2, r3
endif
```

`if` runs the first path for lanes where the condition is true. `else` runs the other path for lanes that were active on entry. `endif` reconverges those lanes.

The emulator stores this state in each wave's control flow stack. Backends map the same behavior to their target's control flow mechanisms.

## Loops

```asm
loop
    iadd r0, r0, r1
    icmp_ge p1, r0, r2
    break p1
endloop
```

`break` removes matching lanes from the loop. `continue` skips the rest of the iteration for matching lanes. `endloop` starts the next iteration for lanes that remain in the loop.

## Predication

An instruction predicate restricts the active lanes. Use `@p1` through `@p3`, or their negated forms, to select lanes. A zero predicate selector with no negation means unpredicated execution. The register `p0` remains writable for comparisons and control-flow operands.

```asm
@p1 device_store r0, r1
```

Only active lanes with `p1` set perform the store. Other lanes leave memory unchanged.

## Wave independence

Branches in one wave cannot change another wave's masks or control flow stacks. Workgroup barriers coordinate waves without sharing their branch state. See [Encoding and execution invariants](/internals/spec-defects/).

The specification requires support for at least 32 nested control flow levels. See [WAVE v0.4](/reference/spec-v04/) for the execution rules and limits.
