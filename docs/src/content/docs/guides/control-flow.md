---
title: Control Flow
description: Structured branching, loops, predicate registers, and divergence behavior in WAVE assembly.
---

WAVE uses structured control flow - every branch and loop has explicit begin and end markers - and handles divergence automatically through per-wave active masks.

## Predicate Registers

Before branching, you need a condition. Compare instructions write their result to one of the four predicate registers (`p0`–`p3`):

```asm
icmp.lt p0, r0, r1    ; p0 = (r0 < r1)  - signed integer compare
ucmp.ge p1, r2, r3    ; p1 = (r2 >= r3) - unsigned integer compare
fcmp.eq p2, r4, r5    ; p2 = (r4 == r5) - float compare
```

### Comparison Conditions

All compare instructions support these conditions:

| Condition | Meaning |
|---|---|
| `eq` | Equal |
| `ne` | Not equal |
| `lt` | Less than |
| `le` | Less than or equal |
| `gt` | Greater than |
| `ge` | Greater than or equal |

Float compares additionally support:
- `ord` - both operands are not NaN (ordered)
- `unord` - at least one operand is NaN (unordered)

## If / Else / Endif

The basic conditional block:

```asm
icmp.lt p0, r0, r1       ; p0 = (r0 < r1)
if p0
  ; executed when p0 is true
  iadd r2, r0, r1
else
  ; executed when p0 is false
  isub r2, r0, r1
endif
```

The `else` block is optional:

```asm
fcmp.gt p0, r0, r1
if p0
  mov_imm r0, 0           ; clamp negative values to zero
endif
```

### What Happens at the Hardware Level

A wave (typically 32 or 64 threads) executes instructions in lockstep. When the threads in a wave disagree on a branch condition - some have `p0 = true`, others `p0 = false` - the wave **diverges**:

1. The hardware pushes the current active mask onto the divergence stack.
2. The `if`-branch executes with only the true-lanes active. False-lanes are masked off (they do not execute, do not write registers, and do not access memory).
3. At `else`, the mask flips: false-lanes become active, true-lanes are masked.
4. At `endif`, the original mask is restored from the stack.

**Both paths always execute** when a wave diverges. If every lane agrees, only the taken path executes - no penalty.

```
Threads:     T0  T1  T2  T3   (4-lane wave for illustration)
Condition:    T   F   T   F

if p0        [T0, --, T2, --]  ← true lanes active
  iadd ...   T0 and T2 execute
else         [--, T1, --, T3]  ← false lanes active
  isub ...   T1 and T3 execute
endif        [T0, T1, T2, T3]  ← all lanes restored
```

## Loops

Loops use `loop` / `endloop` with conditional `break` and `continue`:

```asm
mov_imm r0, 0               ; r0 = i = 0
mov_imm r1, 100             ; r1 = limit

loop
  icmp.ge p0, r0, r1        ; p0 = (i >= 100)
  break p0                   ; exit loop if p0 is true

  ; loop body
  iadd r2, r2, r0            ; accumulate sum

  iadd r0, r0, 1             ; i++
endloop
```

### Break and Continue

Both `break` and `continue` are **predicated** - they take a predicate register and only affect lanes where that predicate is true:

```asm
loop
  ; ... compute some condition ...
  icmp.eq p0, r3, 0
  continue p0               ; skip rest of body for lanes where r3 == 0

  ; only lanes with r3 != 0 reach here
  ; ... expensive computation ...

  icmp.ge p1, r0, r1
  break p1                   ; exit for lanes where i >= limit
endloop
```

When some lanes `break` but others do not, the broken lanes are masked off for all subsequent iterations. The loop continues until every lane has broken or the condition is universally false.

### Loop Divergence

Loops interact with the divergence stack the same way conditionals do:

1. At `loop`, the hardware records the active mask.
2. `break p0` removes the true-lanes from the active set. They are "parked" and will rejoin after `endloop`.
3. `continue p0` temporarily deactivates the true-lanes for the remainder of the current iteration. They rejoin at the top of the next iteration.
4. At `endloop`, if any lanes are still active, execution jumps back to `loop`. If all lanes have broken, execution falls through.

## Nested Control Flow

`if`/`else`/`endif` and `loop`/`endloop` can nest arbitrarily. Each nesting level pushes an additional entry onto the divergence stack.

```asm
loop
  icmp.ge p0, r0, r1
  break p0

  ; Outer condition
  icmp.gt p1, r2, r3
  if p1
    ; Inner condition
    fcmp.lt p2, r4, r5
    if p2
      fadd r6, r6, r4
    else
      fsub r6, r6, r4
    endif
  endif

  iadd r0, r0, 1
endloop
```

At maximum nesting, up to three masks may be stacked (outer loop, outer if, inner if). Deep nesting increases divergence stack pressure and can reduce performance, so keep nesting shallow when possible.

## The Select Instruction

For simple conditional assignments, `select` avoids branching entirely. It picks one of two values based on a predicate - all lanes execute it, no divergence occurs:

```asm
icmp.lt p0, r0, r1
select r2, p0, r3, r4     ; r2 = p0 ? r3 : r4
```

Use `select` instead of `if`/`else`/`endif` when both sides are cheap single-value computations. It is always faster than branching when the bodies are one instruction each.

## Practical Guidelines

**Minimize divergence.** When threads in a wave take different paths, both paths execute sequentially. If your branch splits the wave 50/50, you pay the cost of both sides.

**Prefer `select` over short branches.** A branchless `select` costs one instruction. An `if`/`else`/`endif` with one instruction per side costs the same number of ALU instructions but adds mask manipulation overhead and potential divergence.

**Structure loops so threads exit together.** If most lanes break at the same iteration, only one iteration runs with reduced occupancy. If lanes break at scattered iterations, the loop runs at reduced throughput for many iterations.

**Avoid deep nesting.** Each nesting level adds divergence stack overhead. Flatten conditions with `and`/`or` where possible:

```asm
; Instead of nested ifs:
;   if p0
;     if p1
;       ...
; Use:
and p2, p0, p1
if p2
  ; ...
endif
```

**Use predicated break/continue.** They are more efficient than wrapping the loop body in an `if` block because they directly modify the active mask without pushing a new stack entry.

## Summary

| Construct | Syntax | Divergence cost |
|---|---|---|
| Conditional | `if p / else / endif` | Both paths execute when lanes disagree |
| Loop | `loop / break p / continue p / endloop` | Runs until all lanes exit |
| Branchless select | `select r, p, a, b` | None - always executes in one step |
| Nesting | Arbitrary | Each level adds one divergence stack entry |

Next: [Optimization](/guides/optimization/) - learn how to get the most performance out of your WAVE kernels.
