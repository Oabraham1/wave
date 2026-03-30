---
title: Emulator
description: How wave-emu executes .wbin binaries on the CPU with full SIMT simulation.
---

The WAVE emulator (`wave-emu`) is a CPU-based execution engine that runs `.wbin` binaries without requiring GPU hardware, providing a reference implementation for the WAVE ISA and enabling GPU kernel testing in CI environments.

## Execution Model

The emulator implements a SIMT (Single Instruction, Multiple Threads) execution model that mirrors real GPU hardware behavior.

### Waves and Threads

A kernel launch creates one or more **waves** (the WAVE equivalent of NVIDIA warps or AMD wavefronts). Each wave contains a configurable number of threads that execute in lockstep. The default wave width is 32, matching the most common hardware configuration, but can be set to any value for testing:

```
wave-emu kernel.wbin --wave-width 64   # Test AMD CDNA-style execution
wave-emu kernel.wbin --wave-width 16   # Test narrow wave behavior
```

Threads within a wave share the same program counter when executing uniform control flow. Each thread has its own private register file (32 general-purpose registers + 4 predicate registers) and its own lane index within the wave.

### Workgroup Execution

Waves are grouped into **workgroups**. All waves within a workgroup share access to local memory and can synchronize via barriers. The emulator executes workgroups sequentially (one at a time) to ensure deterministic results, though the execution order of waves within a workgroup is interleaved to expose concurrency bugs.

## Memory Simulation

### Device Memory

Device (global) memory is simulated as a flat byte-addressable buffer allocated on the host heap. All workgroups share the same device memory, matching GPU behavior. The emulator supports:

- **Byte, half-word, word, and double-word** loads and stores (8, 16, 32, 64 bits).
- **Aligned and unaligned access** - unaligned accesses succeed but the emulator emits a diagnostic warning, since many GPU architectures penalize or fault on misaligned access.
- **Out-of-bounds detection** - accesses beyond the allocated buffer size trigger an immediate error with the offending thread ID and program counter.

### Local Memory

Each workgroup receives an independent local (shared) memory allocation sized according to the `.wbin` metadata. Local memory is:

- Visible to all threads in the workgroup but not across workgroups.
- Uninitialized at workgroup start (the emulator fills it with a poison value in debug builds to catch use-before-write bugs).
- Deallocated when the workgroup completes.

### Register File

Each thread maintains a private register file:

- **32 general-purpose registers** (`r0`-`r31`): 32-bit untyped storage. Integer and floating-point values coexist via bitwise reinterpretation.
- **4 predicate registers** (`p0`-`p3`): 1-bit boolean flags used for conditional execution and branch decisions.
- **16 special registers** (read-only): thread ID (`sr_tid_x/y/z`), workgroup ID (`sr_wg_id_x/y/z`), workgroup size (`sr_wg_size_x/y/z`), grid size, wave width, and lane ID.

## Divergence Handling

When threads within a wave take different branch paths, the wave **diverges**. The emulator handles divergence using an active mask and a control flow stack, matching the structured control flow model of the WAVE ISA.

### Active Mask

Every wave maintains an **active mask** - a bitmask where each bit indicates whether the corresponding thread is currently executing. When all threads are active, every bit is set. When a conditional branch causes divergence, the emulator splits execution:

```
                            Active mask: 11111111
                                 │
                            if (r0 > 0)
                           ╱            ╲
               true path                 false path
          Active: 11010010          Active: 00101101
                           ╲            ╱
                            reconverge
                            Active mask: 11111111
```

Instructions execute only for threads whose active mask bit is set. Inactive threads retain their register state but do not read or write memory.

### Control Flow Stack

The emulator maintains a **control flow stack** to track nested divergence. Each stack entry records:

- The active mask before the branch.
- The reconvergence point (the instruction after the `endif` or `endloop`).
- The deferred mask (threads that will execute the else branch or loop exit path).

When entering an `if` block, the emulator pushes the current state, computes the true-mask (threads where the predicate is true), and sets the active mask to the true-mask. When reaching `else`, it swaps to the deferred mask. At `endif`, it pops the stack and restores the full mask.

Loops work similarly: the active mask tracks which threads are still iterating. A `break` instruction clears a thread's active mask bit. When all threads have broken, the loop exits.

## Barrier Synchronization

The `barrier` instruction synchronizes all threads within a workgroup. The emulator implements barriers by:

1. **Suspending the current wave** at the barrier instruction.
2. **Executing other waves** in the workgroup until all waves have reached a barrier.
3. **Resuming all waves** simultaneously past the barrier.

The emulator verifies that all threads in every wave reach the same barrier instruction. If some threads in a wave are inactive due to divergence when a barrier is encountered, the emulator raises a diagnostic error - executing a barrier in divergent control flow is undefined behavior under the WAVE specification.

## Atomic Operations

The emulator implements all WAVE atomic operations with sequential consistency on the CPU. Supported operations:

| Operation | Description |
|---|---|
| `atom_add` | Atomic add (integer) |
| `atom_sub` | Atomic subtract (integer) |
| `atom_and` | Atomic bitwise AND |
| `atom_or` | Atomic bitwise OR |
| `atom_xor` | Atomic bitwise XOR |
| `atom_min` | Atomic minimum |
| `atom_max` | Atomic maximum |
| `atom_exch` | Atomic exchange |
| `atom_cas` | Atomic compare-and-swap |

Each atomic specifies a **scope** (wave, workgroup, device, system) that determines which threads observe the operation's effects. The emulator tracks scoped visibility by maintaining per-scope memory views and flushing writes at fence and barrier points.

## Wave Collective Operations

The emulator supports wave-level collective operations that communicate values across threads within a wave:

- **`wave_shuffle`** - Read the register value from an arbitrary lane in the same wave.
- **`wave_reduce_add/min/max`** - Reduce a value across all active threads in the wave.
- **`wave_ballot`** - Produce a bitmask of which threads have a true predicate.
- **`wave_broadcast`** - Copy a value from one lane to all lanes.

These operations respect the active mask: only active threads participate in reductions and ballots. A `wave_shuffle` that reads from an inactive lane returns an undefined value (the emulator returns zero and emits a warning).

## Debugging Features

### Trace Mode

Trace mode logs every instruction executed by every thread:

```
wave-emu kernel.wbin --trace
```

Output includes the program counter, instruction mnemonic, operand values, and result for each thread:

```
[wave 0, lane 0] PC=0x004  iadd r2, r0, r1  | r0=5 r1=3 → r2=8
[wave 0, lane 1] PC=0x004  iadd r2, r0, r1  | r0=7 r1=3 → r2=10
[wave 0, lane 0] PC=0x008  store [r3], r2    | addr=0x100 val=8
```

Trace output can be filtered by wave index, lane index, or program counter range to reduce volume on large kernels.

### Memory Access Log

The emulator can log all memory accesses (loads, stores, atomics) with timestamps, enabling detection of data races and uncoalesced access patterns:

```
wave-emu kernel.wbin --mem-trace
```

### Breakpoints

The emulator supports program counter breakpoints for interactive debugging:

```
wave-emu kernel.wbin --break 0x010
```

When a breakpoint is hit, the emulator prints the full register state for all threads in the wave and pauses for user input.

### Assertions

Kernels can include `assert` instructions that halt execution with a diagnostic message when a predicate is false. In CI pipelines, a failing assertion causes `wave-emu` to exit with a nonzero status code, enabling standard test harness integration.

**Next:** [ISA Design](/architecture/isa-design/) for the research methodology behind WAVE's 11 hardware-invariant primitives.
