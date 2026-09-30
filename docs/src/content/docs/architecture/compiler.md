---
title: "Compiler internals"
description: "How wave-compiler transforms source kernels into portable .wbin binaries through seven compilation stages."
---

The WAVE compiler, `wave-compiler`, parses a kernel subset of Python, Rust, C++, or TypeScript and emits `.wbin` binaries. Its stages are frontend parsing, HIR construction, MIR lowering, optimization, LIR lowering, register allocation, and binary emission.

## Compilation stages

```
Source kernel (Python / Rust / C++ / TS)
    │
    ▼
┌─────────┐
│ Frontend │   Language-specific parser
└────┬────┘
     │
     ▼
┌─────────┐
│   HIR   │   High-level IR (source structure preserved)
└────┬────┘
     │
     ▼
┌─────────┐
│   MIR   │   SSA-based mid-level IR
└────┬────┘
     │
     ▼
┌─────────────┐
│ Optimization │   DCE, CSE, SCCP, LICM, strength reduction, ...
└──────┬──────┘
     │
     ▼
┌─────────┐
│   LIR   │   Low-level IR with structured control flow
└────┬────┘
     │
     ▼
┌──────────┐
│ RegAlloc │   Interference graph coloring
└────┬─────┘
     │
     ▼
┌──────────┐
│ Emission │   .wbin binary encoding
└──────────┘
```

## Frontend

Each frontend has a dedicated parser and produces the same HIR representation. The parsers support kernel declarations, selected scalar expressions, indexed buffers, structured control flow, and recognized intrinsics. They are not complete implementations of the source languages. Language features such as C++ template instantiation and operator overload resolution are outside this parser model.

## HIR: high-level IR

HIR preserves named variables, `for` and `while` loops, `if/else` branches, and function calls for diagnostics and validation.

HIR performs the following tasks:

- Type checking verifies operand types, including `i32`, `u32`, `f32`, `f16`, `f64`, and `bool`.
- Semantic validation checks the supported kernel subset.
- Special-register binding maps thread indices to registers such as `sr_tid_x` and `sr_wg_id_x`.
- Intrinsic recognition identifies barriers, atomics, and wave operations for lowering.

## MIR: mid-level IR

MIR uses static single assignment form, or SSA. Each value has one definition; phi nodes merge values where control flow joins.

Lowering from HIR to MIR involves:

1. SSA construction gives each definition a distinct value and inserts phi nodes at joins.
2. Control-flow construction converts loops and conditionals into basic blocks and branch edges.
3. Address lowering expresses memory accesses as base addresses and offsets in local or device memory.

Each MIR block contains SSA instructions and ends with a branch, conditional branch, or return.

## Optimization

The MIR pipeline runs until it reaches a fixed point or completes ten iterations.

| Level | Passes |
|-------|--------|
| O0 | No optimization |
| O1 | Private-load forwarding, constant folding, dead-code elimination, CFG simplification |
| O2 | Private-load forwarding, constant propagation, dead-code elimination, common subexpression elimination, loop-invariant code motion, strength reduction, CFG simplification |
| O3 | O2 plus loop unrolling |

Dead-code elimination keeps terminators, side effects, and their transitive dependencies. It removes unused instructions and phi nodes.

Common subexpression elimination reuses typed binary expressions within a block. Private-load forwarding reuses a matching store's value within a block and invalidates it on memory writes or calls.

Constant propagation folds supported integer expressions and prunes constant branches. CFG simplification removes unreachable blocks and merges blocks while preserving phi inputs.

Loop-invariant code motion moves safe computations to a unique preheader. Memory reads, wave operations, and potentially trapping divisions stay in the loop. Strength reduction turns integer multiplication by powers of two into shifts.

Loop unrolling handles single-block loops with at most four statically determined iterations and 128 emitted instructions. It assigns fresh SSA values and preserves loop-carried values and instruction order.

## LIR: low-level IR

LIR converts MIR values into instructions for `.wbin` encoding:

- Values use virtual registers. Register copies replace phi nodes.
- `if/else/endif` and `loop/break/endloop` represent branches and loops.
- Opcodes identify operand types such as `i32` and `f32`.

LIR also inserts any implicit operations required by the ISA, such as converting boolean predicate results into predicate register writes.

## Register allocation

Register allocation maps virtual values to WAVE register indices within a configurable limit of at most 256. Kernel parameters keep their argument indices. Live ranges form an interference graph, and graph coloring assigns indices so overlapping values have different storage.

The compiler returns an error when live values exceed the register limit. It does not insert spills.

The `.wbin` metadata records the WAVE register bound. These indices are still virtual to the vendor compiler. They do not determine machine register usage or GPU occupancy.

## Emission

The emitter walks the register-allocated LIR and encodes each instruction into the `.wbin` binary format:

- The first 32-bit word holds the opcode, two 8-bit register fields, the modifier, and predication bits.
- Instructions that need more operands or an immediate use a second 32-bit word.

See [Binary encoding](/internals/binary-encoding/) for the field layout.

The emitter also produces the binary header containing:

- Kernel entry point name
- Register usage count
- Local memory size requirement
- Workgroup dimensions `x`, `y`, and `z`
- WAVE ISA version

Backends and the emulator read the resulting `.wbin` file.

Next: [Backends](/architecture/backends/) for how each backend translates `.wbin` to vendor-native code.
