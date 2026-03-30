---
title: CLI Tools Reference
description: Command-line reference for wave-asm, wave-dis, wave-compiler, and wave-emu.
---

The WAVE project includes four command-line tools for assembling, disassembling, compiling, and emulating WAVE programs.

## wave-asm

Assembles WAVE assembly source files (`.wave`) into WAVE binary files (`.wbin`).

### Usage

```
wave-asm <input.wave> -o <output.wbin> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `input.wave` | Yes | Input assembly source file. |

### Options

| Option | Description |
|--------|-------------|
| `-o <path>` | Output binary file path. Required. |
| `--dump-hex` | Print the encoded binary as hexadecimal to stdout after assembly. |
| `--dump-ast` | Print the parsed AST to stdout (useful for debugging the assembler). |
| `-v`, `--verbose` | Enable verbose output showing each instruction as it is encoded. |
| `--no-symbols` | Strip symbol table from the output binary. Reduces file size but disables disassembly with labels. |
| `-W <level>` | Set warning level. `0` = suppress all warnings, `1` = default, `2` = treat warnings as errors. |

### Examples

```bash
# Basic assembly
wave-asm vector_add.wave -o vector_add.wbin

# Assemble with hex dump for inspection
wave-asm kernel.wave -o kernel.wbin --dump-hex

# Assemble with all warnings as errors
wave-asm kernel.wave -o kernel.wbin -W 2

# Strip symbols for smaller binary
wave-asm kernel.wave -o kernel.wbin --no-symbols
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Assembly error (syntax error, undefined label, encoding error). |
| `2` | I/O error (file not found, permission denied). |

---

## wave-dis

Disassembles WAVE binary files (`.wbin`) back into human-readable assembly.

### Usage

```
wave-dis <program.wbin> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `program.wbin` | Yes | Input binary file. |

### Options

| Option | Description |
|--------|-------------|
| `-o <path>` | Write disassembly to a file instead of stdout. |
| `--offsets` | Show byte offsets for each instruction (e.g., `0x0010: ADD r0, r1, r2`). |
| `--raw` | Show raw hex encoding alongside each disassembled instruction. |

### Examples

```bash
# Disassemble to stdout
wave-dis program.wbin

# Disassemble to file with offsets
wave-dis program.wbin -o program.wave --offsets

# Show raw instruction encoding
wave-dis program.wbin --raw
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Decode error (invalid opcode, truncated instruction). |
| `2` | I/O error. |

---

## wave-compiler

Compiles high-level kernel source code (Python, Rust, C++, TypeScript) into WAVE binary format.

### Usage

```
wave-compiler <input> -o <output.wbin> --lang <language> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Input source file. |

### Options

| Option | Description |
|--------|-------------|
| `-o <path>` | Output binary file path. Required. |
| `--lang <language>` | Source language. Required. One of: `python`, `rust`, `cpp`, `typescript`. |
| `-O <level>` | Optimization level. `0` = no optimization, `1` = basic (constant folding, dead code elimination), `2` = standard (includes register allocation optimization), `3` = aggressive (includes instruction scheduling and loop unrolling). Default: `1`. |

### Supported Languages

| Language | `--lang` value | File Extensions |
|----------|---------------|-----------------|
| Python | `python` | `.py` |
| Rust | `rust` | `.rs` |
| C++ | `cpp` | `.cpp`, `.cc`, `.cxx` |
| TypeScript | `typescript` | `.ts` |

### Examples

```bash
# Compile a Python kernel
wave-compiler kernel.py -o kernel.wbin --lang python

# Compile with aggressive optimization
wave-compiler matmul.py -o matmul.wbin --lang python -O 3

# Compile a Rust kernel
wave-compiler reduce.rs -o reduce.wbin --lang rust

# Compile a C++ kernel with no optimization (for debugging)
wave-compiler scan.cpp -o scan.wbin --lang cpp -O 0

# Compile a TypeScript kernel
wave-compiler filter.ts -o filter.wbin --lang typescript
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Compilation error (parse error, type error, unsupported construct). |
| `2` | I/O error. |

---

## wave-emu

Emulates WAVE binary programs on the CPU. Simulates the GPU execution model including waves, workgroups, shared memory, and wave operations.

### Usage

```
wave-emu <program.wbin> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `program.wbin` | Yes | Input binary file. |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--grid X,Y,Z` | *(required)* | Grid dimensions. Total number of workgroups = X * Y * Z. |
| `--workgroup X,Y,Z` | *(required)* | Workgroup dimensions. Threads per workgroup = X * Y * Z. |
| `--registers N` | `32` | Number of general-purpose registers per thread. |
| `--local-memory N` | `65536` | Local (shared) memory size in bytes per workgroup. |
| `--device-memory N` | `1048576` | Device (global) memory size in bytes. Default is 1 MB. |
| `--wave-width N` | `32` | Lanes per wave. Must be a power of two. Typical values: 32 (NVIDIA-style) or 64 (AMD-style). |
| `--trace` | *(off)* | Print every executed instruction with register state. Produces large output; use only for small programs. |
| `--stats` | *(off)* | Print execution statistics after completion: total instructions executed, waves dispatched, barriers hit, memory operations. |

### Memory Initialization

The emulator reads device memory contents from buffer files if provided. Use `--buffer` to map host data into device memory:

```bash
wave-emu kernel.wbin --grid 1,1,1 --workgroup 64,1,1 \
  --buffer 0:input.bin --buffer 256:output.bin
```

The `--buffer OFFSET:FILE` option loads raw binary data from `FILE` into device memory starting at byte offset `OFFSET`.

### Examples

```bash
# Basic execution
wave-emu program.wbin --grid 1,1,1 --workgroup 64,1,1

# Multi-workgroup dispatch
wave-emu program.wbin --grid 4,1,1 --workgroup 256,1,1

# 3D grid
wave-emu volume.wbin --grid 8,8,8 --workgroup 4,4,4

# AMD-style wave width with tracing
wave-emu debug.wbin --grid 1,1,1 --workgroup 64,1,1 --wave-width 64 --trace

# Large memory allocation with statistics
wave-emu matmul.wbin --grid 16,16,1 --workgroup 16,16,1 \
  --device-memory 16777216 --local-memory 49152 --stats

# Custom register count
wave-emu kernel.wbin --grid 1,1,1 --workgroup 32,1,1 --registers 16
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Program completed successfully. |
| `1` | Runtime error (out-of-bounds access, invalid instruction, division by zero). |
| `2` | I/O error or invalid arguments. |
| `3` | Timeout (execution exceeded the default limit of 10 billion instructions). |

### Trace Output Format

When `--trace` is enabled, each executed instruction is printed as:

```
[wave W lane L] PC: MNEMONIC operands    | rd=VALUE
```

Example:

```
[wave 0 lane 0] 0x0000: ADD r0, r1, r2    | r0=0x00000003
[wave 0 lane 0] 0x0004: FADD r3, r4, r5   | r3=0x40A00000
[wave 0 lane 0] 0x0008: STORE [r6 + 0x0], r0
```

### Statistics Output

When `--stats` is enabled, a summary is printed after execution:

```
=== Execution Statistics ===
Instructions executed:  12,544
Waves dispatched:       8
Barriers hit:           16
Global loads:           2,048
Global stores:          1,024
Local loads:            4,096
Local stores:           4,096
Atomic operations:      0
Wave operations:        256
Cycles (estimated):     3,200
```
