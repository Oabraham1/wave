---
title: Control Flow
description: How WAVE handles divergent control flow across SIMD lanes using per-wave divergence stacks and structured constructs.
---

WAVE uses structured control flow with per-wave divergence stacks to manage lane divergence, avoiding the complexity of arbitrary jump targets while remaining expressive enough for real GPU workloads.

## The Divergence Problem

GPU threads execute in lockstep groups (waves). When threads within a wave take different branches of an `if` statement, the hardware must track which lanes are active and which are masked off. This is divergence, and every GPU vendor handles it differently at the hardware level.

The challenge for a portable ISA: define divergence behavior precisely enough that programs are correct on all targets, without prescribing a mechanism that only one vendor implements.

## Structured Control Flow

WAVE mandates structured control flow constructs rather than arbitrary branch/jump instructions:

```
if p0                ; push active mask, mask lanes where p0 is false
  ; ... then-body (only p0-true lanes execute)
else                 ; flip mask
  ; ... else-body (only p0-false lanes execute)
endif                ; pop mask, restore all lanes

loop                 ; push loop mask
  ; ... loop body
  break p1           ; lanes where p1 is true exit the loop
endloop              ; branch back to loop header for remaining lanes
```

### Why structured, not arbitrary jumps?

1. **All four vendors use structured divergence internally.** Even NVIDIA's per-thread program counter reconverges at structured control flow boundaries. Arbitrary jumps would require a reconvergence analysis that the hardware already performs for structured constructs.

2. **Correctness is provable.** Structured control flow guarantees that every divergent region has a well-defined reconvergence point. With arbitrary jumps, the backend would need to insert reconvergence barriers --- a solved but fragile problem.

3. **Backend mapping is direct.** Each structured construct maps to a small, well-understood sequence on every target (see [Backend Mapping](/internals/backend-mapping/)).

## Per-Wave Divergence Stacks

Each wave maintains its own divergence stack. The stack tracks:

- **Active mask**: which lanes are currently executing.
- **Reconvergence point**: where masked-off lanes will rejoin.

When an `if` is encountered, the current active mask is pushed onto the stack, and the new mask reflects the predicate evaluation. At `else`, the mask is complemented (relative to the pushed mask). At `endif`, the original mask is popped and restored.

### The Shared-Stack Defect (v0.1)

The v0.1 specification used a single divergence stack shared across all waves in a workgroup. This caused a deadlock scenario:

1. Wave A pushes a mask and enters a divergent region.
2. Wave B, at a different program counter, hits a barrier and waits.
3. Wave A also reaches the barrier, but the shared stack now contains stale data from Wave B's earlier divergence.
4. Neither wave can proceed --- the stack state is corrupted.

The fix was straightforward: each wave gets its own divergence stack. This matches every vendor's implementation (see below) and eliminates cross-wave interference entirely. Full details in [Spec Defects](/internals/spec-defects/).

## Vendor Divergence Mechanisms

| Vendor | Mechanism | WAVE Mapping |
|--------|-----------|--------------|
| **NVIDIA** | Per-thread program counter with convergence barriers (`bra.uni`, `bar.sync`). Hardware tracks per-thread PCs and reconverges at sync points. | Structured constructs map to predicated branches with implicit reconvergence. |
| **AMD** | Compiler-managed `EXEC` mask. The `s_cbranch` instructions modify EXEC; the compiler inserts `s_or_b64` to restore masks. | `if`/`else`/`endif` map directly to EXEC mask manipulation sequences. |
| **Intel** | Predicated SIMD execution. The EU evaluates all lanes but applies a channel enable mask per instruction. | Structured constructs map to predicate register updates and predicated instruction sequences. |
| **Apple** | Hardware divergence stack stored in register `r0l`. The GPU pushes/pops masks automatically for structured constructs. | Near 1:1 mapping --- Apple's hardware natively supports the same structured model. |

## Predicated Execution

WAVE provides four predicate registers (`p0`--`p3`) selected by the 2-bit predicate field in the instruction encoding. The `pred_neg` bit inverts the condition. Any instruction can be predicated:

```
cmp.lt p0, r0, r1   ; set p0 where r0 < r1
if p0
  iadd r2, r2, r3   ; only executes in lanes where r0 < r1
endif
```

Predicates interact with the divergence stack: an `if` statement evaluates its predicate and intersects the result with the current active mask to produce the new mask.

## Nesting

Structured constructs can nest to arbitrary depth. Each nesting level pushes one entry onto the divergence stack. In practice, kernel divergence depth rarely exceeds 4--6 levels, and the stack is bounded by implementation-defined limits exposed through the WAVE runtime query interface.
