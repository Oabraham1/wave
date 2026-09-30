---
title: "Emulator"
description: "Execute WAVE binaries on the CPU with wave state, memory checks, and tracing."
---

`wave-emu` executes `.wbin` files on the CPU. It provides functional tests for kernels without modeling GPU timing.

## Execution model

Each wave has a program counter, active mask, and control-flow stacks. Each thread has registers and a lane index.

### Waves and threads

The default wave width is 32. Set another width to test kernels whose lane operations support it:

```bash
wave-emu kernel.wbin --wave-width 64
wave-emu kernel.wbin --wave-width 16
```

The register count defaults to 32. A nonzero count in kernel metadata overrides this setting. Each thread also has four predicate registers and access to sixteen special-register values.

### Workgroup execution

Workgroups run sequentially. Within a workgroup, the scheduler selects ready waves in round-robin order. Suspended and halted waves do not execute. This schedule makes a run repeatable; it does not explore every legal GPU interleaving.

## Memory simulation

### Device memory

All workgroups share a byte-addressed host buffer. It starts at zero. Loads and stores check the requested byte range and return an error on an out-of-bounds access.

Accesses use little-endian byte operations. The emulator does not diagnose every alignment requirement imposed by a GPU target.

### Local memory

Each workgroup gets a separate zero-initialized buffer. A nonzero local-memory size in kernel metadata overrides the configured size. Kernels must still initialize memory before reading it when the ISA requires them to do so; the emulator's initialization is not a target guarantee.

### Register file

- General-purpose registers hold untyped 32-bit values.
- Predicate registers `p0` through `p3` hold boolean values.
- Special registers expose thread, wave, workgroup, and grid identifiers and dimensions.

See [Register model](/internals/register-model/) for register addressing and wide values.

## Divergence handling

### Active mask

Only active lanes execute an instruction. Predication can restrict this set further. Inactive lanes leave destination registers and memory unchanged.

### Control flow stack

An `if` partitions the incoming active lanes by its condition. `else` selects the other path, and `endif` restores the participating lanes. Loop state tracks lanes that break or continue independently of other waves.

See [Control flow](/internals/control-flow/) for examples.

## Barrier synchronization

At a workgroup barrier, the current wave suspends. The scheduler runs other ready waves until all workgroup waves arrive, then releases them. A workgroup with suspended waves and no ready wave can report deadlock.

Programs must obey the specification's barrier participation rules. Successful emulator completion alone does not prove that every lane reaches the same barrier correctly.

## Atomic operations

The emulator executes atomic read-modify-write operations on local or device memory:

| Operation | Effect |
|---|---|
| Add and subtract | Update an integer value |
| AND, OR, and XOR | Update selected bits |
| Minimum and maximum | Select a value by comparison |
| Exchange | Replace the value |
| Compare-and-swap | Replace a matching value and return the previous value |

These operations execute within the emulator's sequential schedule. It does not maintain separate cache views for each memory scope or simulate weak memory ordering.

## Wave collective operations

Shuffle exchanges register values across lanes. Reductions combine lane values, ballots collect predicates, and broadcast selects one source lane. The implementation reads shuffle sources before writing destinations, so overlapping registers preserve cross-lane dependencies.

Kernels must satisfy the [specification's](/reference/spec-v04/) active-lane and source-lane requirements.

## Debugging features

### Trace mode

```bash
wave-emu kernel.wbin --trace
```

Trace records identify the workgroup, wave, program counter, and instruction. Use small inputs to keep the output readable.

### Memory access log

`--stats` reports memory-operation counts and transferred bytes. `--dump-memory START:END` prints a memory range after execution. These outputs do not constitute a data-race detector.

### Breakpoints

The CLI supports instruction limits and post-execution dumps rather than interactive breakpoints. `--max-instructions N` stops a run after the configured instruction budget; `0` removes the limit.

### Assertions

A decoding, memory, control-flow, or execution-limit error causes the CLI to fail. Test harnesses can check its exit status and compare the output dumps with an independent reference.

See [CLI tools](/reference/cli-tools/) for all options.
