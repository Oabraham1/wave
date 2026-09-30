---
title: "Control flow"
description: "Predicates, branches, loops, and divergence in WAVE assembly."
---

WAVE branches and loops use explicit begin and end markers. An active mask determines which lanes execute each instruction.

## Predicate registers

Comparisons write to `p0` through `p3`:

```asm
icmp_lt p0, r0, r1    ; signed comparison
ucmp_le p1, r2, r3    ; unsigned comparison
fcmp_eq p2, r4, r5    ; floating-point comparison
```

### Comparison conditions

| Suffix | Meaning |
|---|---|
| `eq` | Equal |
| `ne` | Not equal |
| `lt` | Less than |
| `le` | Less than or equal |
| `gt` | Greater than |
| `ge` | Greater than or equal |

Signed integer comparisons support all six suffixes. Unsigned comparisons use `ucmp_lt` and `ucmp_le`; reverse the operands for greater-than comparisons. Floating-point comparisons support `eq`, `ne`, `lt`, `le`, and `gt`. They also support `ord` when neither operand is NaN and `unord` when at least one is NaN.

## If / else / endif

```asm
icmp_lt p0, r0, r1
if p0
    iadd r2, r0, r1
else
    isub r2, r0, r1
endif
```

The `else` block is optional. This example clamps negative floats to zero:

```asm
mov_imm r1, 0
fcmp_lt p0, r0, r1
if p0
    mov_imm r0, 0
endif
```

### What happens at the hardware level

A wave diverges when its lanes disagree on a branch condition. The execution model applies a mask to each branch:

1. `if` records the entering mask and activates lanes whose predicate is true.
2. `else` activates the entering lanes whose predicate was false.
3. `endif` restores the entering mask, subject to lanes that exited or continued a surrounding loop.

Both paths contain work when lanes disagree. A uniform branch needs only the selected path. The backend determines how these masks map to hardware instructions.

```text
Lane:         0   1   2   3
Predicate:    T   F   T   F
if:          [0,  -,  2,  -]
else:        [-,  1,  -,  3]
endif:       [0,  1,  2,  3]
```

## Loops

Use `loop` and `endloop` with a predicated exit:

```asm
mov_imm r0, 0
mov_imm r1, 100
mov_imm r2, 0
mov_imm r3, 1
loop
    icmp_ge p0, r0, r1
    break p0
    iadd r2, r2, r0
    iadd r0, r0, r3
endloop
```

This loop sums the integers from 0 through 99.

### Break and continue

`break p0` exits the loop for lanes where `p0` is true. `continue p0` skips the rest of the current iteration for those lanes. Advance loop state before a `continue` that would skip the update.

```asm
mov_imm r0, 0
mov_imm r1, 100
mov_imm r2, 0
mov_imm r3, 1
mov_imm r4, 0
loop
    icmp_ge p0, r0, r1
    break p0
    iadd r0, r0, r3
    and r5, r0, r3
    icmp_eq p1, r5, r4
    continue p1
    iadd r2, r2, r0
endloop
```

This loop adds the odd integers from 1 through 99.

### Loop divergence

Lanes that break stay inactive for subsequent iterations and rejoin after the loop. Lanes that continue rejoin at the next iteration. The loop finishes when all participating lanes have exited.

### Nested control flow

Branches and loops can nest. Each level needs its matching closing instruction:

```asm
icmp_lt p0, r0, r1
if p0
    fcmp_lt p1, r2, r3
    if p1
        fadd r4, r2, r3
    else
        fsub r4, r2, r3
    endif
endif
```

## The select instruction

`select` chooses between register values without branching:

```asm
icmp_lt p0, r0, r1
select r2, p0, r3, r4    ; choose r3 when true, otherwise r4
```

Use it when both input values are available and computing them unconditionally is safe. Whether it is faster than a branch depends on the target and surrounding code.

## Practical guidelines

- Group work so neighboring lanes tend to follow the same branches.
- Keep loop iteration counts similar across lanes when the algorithm permits it.
- Use `select` for short assignments with safe unconditional inputs.
- Keep barriers outside divergent branches unless every workgroup thread reaches the same barrier.
- Measure changes on the target GPU. Source instruction counts do not determine execution time.

## Summary

| Construct | Syntax | Lane behavior |
|---|---|---|
| Conditional | `if p / else / endif` | Select lanes by predicate |
| Loop | `loop / break p / continue p / endloop` | Run until participating lanes exit |
| Select | `select r, p, a, b` | Choose a value without changing the active mask |
| Nesting | Matched blocks | Preserve the surrounding control state |

Next: [Optimization](/guides/optimization/).
