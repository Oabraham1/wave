# wave-emu

The reference CPU emulator for the WAVE ISA.

## Overview

wave-emu runs `.wbin` files on the CPU. It implements:

- **Threads, waves, and workgroups**: Configurable wave width, with a default of 32
- **Memory**: Global device memory and workgroup-local memory
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
- `control_flow.rs` - Structured control flow, if/else/loop, handling
- `barrier.rs` - Barrier synchronization
- `scheduler.rs` - Round-robin wave scheduling
- `shuffle.rs` - Wave shuffle and collective operations
- `stats.rs` - Execution statistics collection

## License

Copyright © 2026 Ojima Abraham. All rights reserved.
Licensed under the Apache License, Version 2.0.

## Memory-read benchmark

```bash
cargo bench --manifest-path wave-emu/Cargo.toml --bench device_memory
```

Run this from the repository root. It checks read contents at four buffer sizes and reports median latency and throughput across seven rounds. Each read returns an owned buffer.
