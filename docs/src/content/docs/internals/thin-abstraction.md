---
title: Thin abstraction principle
description: WAVE defines instruction behavior and leaves implementation details to each GPU vendor.
---

WAVE defines instruction behavior. Each GPU vendor chooses how to implement it.

## Hardware parameters

Programs query values that vary by target:

| Parameter | WAVE contract |
|---|---|
| Wave width | Query the target width |
| Register file size | Query the hardware limit and declare kernel usage |
| Workgroup size | Query the maximum dimensions |
| Local memory | Query the available capacity |
| Memory ordering | Use defined scopes and acquire/release semantics |
| Divergence | Use structured branches and loops |

The GPU schedules threads and executes memory operations. Backends translate WAVE instructions to vendor code. They can specialize that code using the target's parameters.

## Wave width

NVIDIA uses 32-wide warps. AMD supports 32-wide and 64-wide wavefronts. Intel supports several subgroup widths. Apple uses 32-wide SIMD groups.

Query the width when a kernel depends on it:

```asm
mov_sr r0, sr_wave_width
```

The backend can specialize loops after it knows the wave width.

## Divergence

PTX uses branches and predicates. HIP, SYCL, and Metal backends emit structured source control flow for their vendor compilers.

WAVE specifies which lanes execute each branch and where they reconverge. Backends implement `if`/`else`/`endif` and `loop`/`break`/`endloop` with the target's control flow mechanism. See [Control flow](/internals/control-flow/).

## Memory ordering

WAVE defines wave, workgroup, device, and system scopes with acquire/release ordering.

PTX uses scoped `membar` instructions, HIP uses thread-fence functions, and SYCL uses `atomic_fence`. Metal uses `atomic_thread_fence` and rejects system scope. Backends preserve the requested ordering and participation requirements. See [Memory scoping](/internals/memory-scoping/).

## Costs

The vendor compiler selects machine instructions, allocates registers, and schedules the generated program. Workgroup size and memory layout also affect execution cost.

WAVE does not expose cache line sizes. Programs use scoped ordering; the backend supplies the required cache operations.

## Related pages

- [Binary encoding](/internals/binary-encoding/)
- [Register model](/internals/register-model/)
- [Occupancy equation](/internals/occupancy-equation/)
- [Backend mapping](/internals/backend-mapping/)
