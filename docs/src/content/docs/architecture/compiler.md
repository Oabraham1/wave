---
title: Compiler Internals
description: How wave-compiler transforms source kernels into portable .wbin binaries through seven compilation stages.
---

The WAVE compiler (`wave-compiler`) transforms GPU kernels written in Python, Rust, C++, or TypeScript into hardware-invariant `.wbin` binaries through a seven-stage pipeline: frontend parsing, HIR construction, MIR lowering, optimization, LIR lowering, register allocation, and binary emission.

## Compilation Stages

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
│ RegAlloc │   Chaitin-Briggs graph coloring
└────┬─────┘
     │
     ▼
┌──────────┐
│ Emission │   .wbin binary encoding
└──────────┘
```

## Frontend

The frontend is the only language-specific component in the compiler. Each supported language has a dedicated parser that transforms kernel source into HIR:

- **Python frontend** - Parses decorated functions (`@wave.kernel`), resolves type annotations, and handles NumPy-style indexing patterns.
- **Rust frontend** - Parses functions annotated with `#[wave::kernel]`, resolves Rust types, and lowers pattern matching into conditional branches.
- **C++ frontend** - Parses functions with `[[wave::kernel]]`, handles template instantiations, and resolves operator overloads.
- **TypeScript frontend** - Parses decorated functions, resolves TypedArray types, and handles JavaScript-specific numeric semantics (all numbers are f64 by default, requiring explicit narrowing).

Every frontend produces the same HIR representation. Once a kernel enters HIR, the language it was written in is irrelevant to every subsequent stage.

## HIR (High-Level IR)

HIR preserves the source-level structure of the kernel: named variables, structured loops (`for`, `while`), conditionals (`if/else`), and function calls. This makes HIR suitable for source-level diagnostics and early validation.

HIR performs the following tasks:

- **Type checking** - Verifies that all operations have compatible operand types. WAVE supports `i32`, `u32`, `f32`, `f16`, `f64`, and `bool` as first-class types.
- **GPU semantic validation** - Ensures the kernel does not use unsupported constructs (recursion, dynamic dispatch, heap allocation).
- **Special register binding** - Maps language-level thread index expressions to WAVE special registers (`sr_tid_x`, `sr_wg_id_x`, etc.).
- **Intrinsic recognition** - Identifies calls to WAVE intrinsics (barriers, atomics, wave operations) and tags them for later lowering.

## MIR (Mid-Level IR)

MIR is the primary IR for analysis and optimization. It uses Static Single Assignment (SSA) form, where every variable is assigned exactly once and phi nodes merge values at control flow join points.

Lowering from HIR to MIR involves:

1. **SSA construction** - Variables become versioned definitions (`x.0`, `x.1`, ...). Phi nodes are inserted at dominance frontiers.
2. **Control flow graph construction** - Structured loops and conditionals are decomposed into basic blocks with explicit branch edges.
3. **Address lowering** - Memory accesses are decomposed into base + offset calculations with explicit address space annotations (local vs. device).

MIR basic blocks contain a linear sequence of SSA instructions, each with a typed result and zero or more operands. The last instruction in each block is a terminator (branch, conditional branch, or return).

## Optimization

The optimization pipeline runs on MIR and is controlled by the optimization level flag:

| Level | Behavior |
|---|---|
| **O0** | No optimizations. Direct lowering from MIR to LIR. Used for debugging. |
| **O1** | Lightweight passes: DCE, CSE, CFG simplification. Fast compilation. |
| **O2** | Full optimization: all O1 passes plus SCCP, LICM, strength reduction, mem2reg. Default level. |
| **O3** | Aggressive: all O2 passes plus loop unrolling, aggressive inlining. May increase register pressure. |

### Individual Passes

**Dead Code Elimination (DCE)** removes instructions whose results are never used. Starting from terminators and side-effecting instructions, it marks all transitively-used instructions as live. Everything unmarked is deleted.

**Common Subexpression Elimination (CSE)** identifies instructions with identical opcodes and operands within the same basic block (local CSE) or across dominating blocks (global CSE). Redundant computations are replaced with references to the first occurrence.

**Sparse Conditional Constant Propagation (SCCP)** combines constant propagation with dead branch elimination. It tracks lattice values (top/constant/bottom) for every SSA value and evaluates branches on constant conditions to prune unreachable blocks.

**Loop-Invariant Code Motion (LICM)** hoists instructions out of loops when their operands are defined outside the loop and the instruction has no side effects. Requires loop detection and dominance analysis as prerequisites.

**Strength Reduction** replaces expensive operations with cheaper equivalents: multiplication by powers of two becomes left shift, division by constants becomes multiply-and-shift sequences, and modulo by powers of two becomes bitwise AND.

**mem2reg** promotes stack-allocated local variables to SSA registers when the variable's address is never taken. This is critical for languages like C++ where local variables are stack-allocated by default.

**Loop Unrolling** (O3 only) duplicates the loop body a configurable number of times to reduce branch overhead and expose more ILP. The unroll factor is limited by register pressure heuristics to avoid excessive spilling.

**CFG Simplification** merges basic blocks with a single predecessor and successor, removes empty blocks, and eliminates unconditional branches to the next sequential block.

## LIR (Low-Level IR)

LIR lowers MIR out of SSA form and into a representation that maps closely to the `.wbin` instruction encoding. Key differences from MIR:

- **No SSA** - Values are assigned to virtual registers that may be overwritten. Phi nodes are replaced with register copies inserted at predecessor block exits.
- **Structured control flow** - Branch targets are expressed as structured `if/else/endif` and `loop/break/endloop` constructs rather than arbitrary basic block edges. This matches the `.wbin` encoding and simplifies backend translation.
- **Explicit type annotations** - Every instruction carries its type (i32, f32, etc.) as part of the opcode, matching the binary encoding.

LIR also inserts any implicit operations required by the ISA, such as converting boolean predicate results into predicate register writes.

## Register Allocation

Register allocation assigns virtual registers to the 32 physical general-purpose registers (`r0`-`r31`) and 4 predicate registers (`p0`-`p3`) defined by the WAVE ISA.

The allocator uses **Chaitin-Briggs graph coloring**:

1. **Build** - Construct an interference graph where each virtual register is a node and edges connect registers that are simultaneously live.
2. **Coalesce** - Merge nodes connected by copy instructions when they do not interfere. This eliminates unnecessary register-to-register moves.
3. **Simplify** - Iteratively remove nodes with degree less than the number of physical registers, pushing them onto a stack.
4. **Spill** - If no node can be simplified, select a spill candidate using a cost heuristic (prioritizing infrequently-used registers in inner loops). Insert load/store instructions to local memory for the spilled register and restart.
5. **Select** - Pop nodes from the stack and assign physical registers, choosing colors that avoid all neighbors.

Register pressure directly impacts GPU occupancy. The compiler reports the register count per kernel in the `.wbin` metadata, which backends use for occupancy calculations via the universal equation `O = floor(F / (R * W * w))`.

## Emission

The emitter walks the register-allocated LIR and encodes each instruction into the `.wbin` binary format:

- **Base instructions (32-bit)** encode: opcode (8 bits), destination register (5 bits), source register 1 (5 bits), source register 2 (5 bits), and modifier flags (9 bits).
- **Extended instructions (64-bit)** encode: opcode (8 bits), destination register (5 bits), three source registers (15 bits), immediate value (24 bits), and modifier flags (12 bits).

The emitter also produces the binary header containing:

- Kernel entry point name
- Register usage count (general-purpose and predicate)
- Local memory size requirement
- Workgroup size hints (x, y, z)
- WAVE ISA version

The output is a self-contained `.wbin` file ready for backend translation or emulator execution.

**Next:** [Backends](/architecture/backends/) for how each backend translates `.wbin` to vendor-native code.
