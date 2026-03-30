---
title: Thin Abstraction Principle
description: WAVE's core design philosophy of defining what hardware must do, not how it does it, and why this enables cross-vendor portability.
---

WAVE follows a thin abstraction principle --- "define what hardware must do, not how" --- that trades compile-time certainty for runtime flexibility, enabling a single ISA to target fundamentally different GPU architectures.

## The Principle

A thin abstraction specifies **behavior** (what happens) without specifying **mechanism** (how it happens). Applied to a GPU ISA, this means:

| Instead of... | WAVE does... |
|--------------|-------------|
| Prescribing wave width W=32 | Querying wave width at runtime |
| Exposing divergence stack operations | Defining structured control flow semantics |
| Specifying cache line sizes and policies | Defining memory scopes and ordering |
| Dictating workgroup tile sizes | Querying maximum workgroup dimensions |
| Fixing register file size | Recording register count in metadata |

Each "instead of" row would lock WAVE to one vendor's design. Each "WAVE does" row leaves the hardware parameter free to vary.

## Why "Thin"?

The abstraction is thin in the sense that it adds minimal indirection between the programmer's intent and the hardware's execution:

- **No virtual machine**: WAVE is not a bytecode that gets interpreted. It is a binary format that gets lowered to native instructions by a thin backend pass.
- **No runtime scheduler**: WAVE does not manage thread scheduling. The hardware scheduler runs as designed.
- **No memory virtualization**: WAVE memory operations map directly to hardware load/store units. Scopes map to hardware coherence domains.

The only runtime overhead is the query mechanism (returning hardware parameters like wave width and register file size), which is a constant-time lookup, not an ongoing cost.

## Historical Precedent: ARM

The same principle made the ARM ISA successful across a vast range of implementations. ARM defines instruction behavior (what an ADD does) without prescribing microarchitecture (pipeline depth, cache sizes, out-of-order execution). This allowed ARM to scale from embedded microcontrollers to server processors.

WAVE applies this at the GPU level: it defines what a barrier does (synchronize all waves in a workgroup) without prescribing how (hardware counter, scoreboard, gateway message). The result is the same kind of implementation freedom that ARM licensees enjoy.

## Concrete Examples

### Wave Width

NVIDIA uses 32-wide warps. AMD supports both 32-wide and 64-wide wavefronts. Intel uses 8-wide or 16-wide sub-groups. Apple uses 32-wide SIMD-groups.

A thick abstraction would pick one width and emulate it everywhere --- forcing NVIDIA to run two 16-wide groups or Intel to pad 16-wide groups to 32. Both waste hardware resources.

WAVE's approach:

```
; Query wave width at runtime
query r0, WAVE_WIDTH    ; r0 = 32 on NVIDIA, 64 on AMD, 16 on Intel, 32 on Apple

; Use it to compute iteration counts, memory offsets, etc.
idiv r1, r_total, r0    ; iterations = total_elements / wave_width
```

The program adapts to the hardware rather than the hardware adapting to the program.

### Divergence

NVIDIA manages divergence with per-thread program counters and convergence barriers. AMD uses compiler-managed EXEC masks. Intel uses predicated SIMD channels. Apple has a hardware stack in `r0l`.

A thick abstraction would expose one mechanism (e.g., explicit mask manipulation) and require three backends to emulate it. WAVE instead defines structured control flow (`if`/`else`/`endif`, `loop`/`break`/`endloop`) with specified behavior (which lanes execute, where they reconverge). Each backend implements this using its native mechanism. See [Control Flow](/internals/control-flow/) for details.

### Memory Scopes

NVIDIA provides `membar.cta`, `membar.gpu`, `membar.sys`. AMD uses `s_waitcnt` with different counter types. Intel uses typed fences (`fence.slm`, `fence.ugm`). Apple uses `threadgroup_barrier` with memory flag arguments.

WAVE defines four scope levels (wave, workgroup, device, system) and acquire/release ordering. Each scope maps to the vendor's native fence at the appropriate coherence boundary. See [Memory Scoping](/internals/memory-scoping/) for the full mapping.

### Tile Sizes and Workgroup Dimensions

Different GPUs have different optimal tile sizes based on their local memory capacity, register file size, and cache line width. WAVE does not prescribe tile sizes. Instead, it exposes queries for maximum workgroup dimensions and local memory size, allowing the compiler or runtime to select optimal tile sizes for the target.

## The Trade-off

Thin abstraction is not free. By deferring hardware parameters to runtime, WAVE gives up some compile-time optimization opportunities:

- **Loop unrolling**: A compiler that knows W=32 can unroll wave-width loops at compile time. WAVE programs must either use runtime loop bounds or rely on the backend to specialize after the wave width is known.
- **Register allocation**: Optimal register allocation depends on the target's register file size, which is a runtime parameter. The WAVE compiler allocates conservatively; the backend can re-allocate if needed.
- **Memory layout**: Optimal data layout depends on cache line sizes, which WAVE does not expose. Programs use scope-based ordering, and the backend inserts cache management instructions as needed.

These trade-offs are acceptable because:

1. **Backend specialization is cheap**: The backend pass that lowers WAVE to native code runs once per target. It has full knowledge of hardware parameters and can perform target-specific optimizations.
2. **The alternative is worse**: A thick abstraction that fixes parameters achieves optimal code for one architecture and suboptimal code for the other three.
3. **Runtime queries are constant-time**: There is no ongoing overhead; parameters are read once during kernel setup.

## Relationship to Other Design Decisions

The thin abstraction principle is not an isolated choice --- it permeates every aspect of WAVE's design:

- [Binary Encoding](/internals/binary-encoding/): The scope field in every instruction is a thin abstraction over vendor-specific coherence mechanisms.
- [Register Model](/internals/register-model/): Untyped registers abstract over vendor-specific ALU pipeline details.
- [Occupancy Equation](/internals/occupancy-equation/): The equation uses hardware parameters as variables, not constants.
- [Backend Mapping](/internals/backend-mapping/): Each WAVE primitive maps to a small, vendor-specific instruction sequence --- the "thin" layer between portable and native code.
