# wave-emu

A reference CPU-based emulator for the WAVE (Wide Architecture Virtual Encoding) ISA.

## Overview

wave-emu executes WAVE binary programs (.wbin files) and simulates the GPU-like execution model including:

- **Threads, Waves, and Workgroups**: Full SIMT execution model with configurable wave width (default 32)
- **Memory Hierarchy**: Device memory (global) and local memory (per-workgroup)
- **Divergence Handling**: Structured control flow with active mask tracking
- **Wave Operations**: Shuffle, broadcast, ballot, prefix sum, reductions
- **Barriers**: Workgroup-level synchronization
- **Atomics**: Local and device memory atomic operations

## Building

```bash
cargo build --release
```

## Usage

```
wave-emu [OPTIONS] <BINARY>

Arguments:
  <BINARY>  WBIN binary file to execute

Options:
      --grid <X,Y,Z>             Grid dimensions [default: 1,1,1]
      --workgroup <X,Y,Z>        Workgroup dimensions [default: 32,1,1]
      --registers <REGISTERS>    Registers per thread [default: 32]
      --local-memory <BYTES>     Local memory size [default: 16384]
      --device-memory <BYTES>    Device memory size [default: 1048576]
      --arg <OFFSET:FILE>        Load file into device memory at offset
      --dump-regs                Print register state after execution
      --dump-memory <START:END>  Hex dump device memory range after execution
      --stats                    Print execution statistics
      --trace                    Print each instruction as it executes
      --wave-width <WAVE_WIDTH>  Wave width [default: 32]
```

## Examples

Execute a WAVE binary:
```bash
wave-emu program.wbin
```

Execute with tracing and statistics:
```bash
wave-emu program.wbin --trace --stats
```

Run with a 2x2 grid of 64-thread workgroups:
```bash
wave-emu program.wbin --grid 2,2,1 --workgroup 64,1,1
```

Load input data and dump output:
```bash
wave-emu program.wbin --arg 0:input.bin --dump-memory 0x1000:0x2000
```

## Architecture

- `lib.rs` - Public API and Emulator struct
- `core.rs` - Core simulation engine managing workgroup execution
- `executor.rs` - Instruction execution dispatcher
- `decoder.rs` - Binary instruction decoder for WBIN format
- `wave.rs` - Wave state management
- `thread.rs` - Per-thread register and predicate state
- `memory.rs` - Device and local memory implementation
- `control_flow.rs` - Structured control flow (if/else/loop) handling
- `barrier.rs` - Barrier synchronization
- `scheduler.rs` - Round-robin wave scheduling
- `shuffle.rs` - Wave shuffle and collective operations
- `stats.rs` - Execution statistics collection

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.
Licensed under the Apache License, Version 2.0.
