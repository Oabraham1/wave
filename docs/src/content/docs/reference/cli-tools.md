---
title: "CLI tools reference"
description: "Assemble, disassemble, compile, and emulate WAVE programs."
---

These tools read source or `.wbin` files. Each accepts `--help` for its command-line options.

## wave-asm

Assemble a `.wave` source file into `.wbin`.

### Usage

```text
wave-asm <input.wave> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `input.wave` | Assembly source file |

### Options

| Option | Description |
|---|---|
| `-o`, `--output <path>` | Output path; defaults to the input path with a `.wbin` extension |
| `--dump-hex` | Print the binary as hexadecimal instead of writing it |
| `--dump-ast` | Parse and print the AST instead of assembling a binary |
| `--verbose` | Print the assembled byte count and output path |
| `--no-symbols` | Strip the symbol table |
| `-W`, `--warn <level>` | Warning policy: `all`, `none`, or `error`; default `all` |

### Examples

```bash
wave-asm vector_add.wave -o vector_add.wbin
wave-asm kernel.wave --dump-hex
wave-asm kernel.wave -o kernel.wbin -W error
wave-asm kernel.wave -o kernel.wbin --no-symbols
```

### Exit codes

`0` indicates success, `1` indicates an assembly or I/O failure, and `2` indicates invalid CLI arguments.

## wave-dis

Disassemble a `.wbin` file into assembly text.

### Usage

```text
wave-dis <program.wbin> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `program.wbin` | Binary file |

### Options

| Option | Description |
|---|---|
| `-o`, `--output <path>` | Write to a file instead of stdout |
| `--offsets` | Include instruction byte offsets |
| `--raw` | Include hexadecimal instruction words |
| `--no-directives` | Omit kernel, register, and end directives |

### Examples

```bash
wave-dis program.wbin
wave-dis program.wbin -o program.wave --offsets
wave-dis program.wbin --raw
```

### Exit codes

`0` indicates success, `1` indicates decoding or I/O failure, and `2` indicates invalid CLI arguments.

## wave-compiler

Compile a supported Python, Rust, C++, or TypeScript kernel into `.wbin`.

### Usage

```text
wave-compiler <input> -o <output.wbin> --lang <language> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `input` | Source file |

### Options

| Option | Description |
|---|---|
| `-o`, `--output <path>` | Required output path |
| `--lang <language>` | Required source language |
| `-O`, `--opt-level <level>` | Optimization level `0` through `3`; default `0` |
| `--max-registers N` | Register limit; default `256` |
| `--wave-width N` | Wave width; default `32` |
| `--dump-hir` | Print HIR to stderr |
| `--dump-mir` | Print MIR to stderr |
| `--dump-lir` | Print LIR to stderr |

See [Compiler optimization](/architecture/compiler/#optimization) for the passes at each level.

### Supported languages

| Language | `--lang` value | File extensions |
|---|---|---|
| Python | `python` | `.py` |
| Rust | `rust` | `.rs` |
| C++ | `cpp` | `.cpp`, `.cc`, `.cxx` |
| TypeScript | `typescript` | `.ts` |

### Examples

```bash
wave-compiler kernel.py -o kernel.wbin --lang python
wave-compiler matmul.py -o matmul.wbin --lang python -O 3
wave-compiler reduce.rs -o reduce.wbin --lang rust
wave-compiler scan.cpp -o scan.wbin --lang cpp -O 0
wave-compiler filter.ts -o filter.wbin --lang typescript
```

### Exit codes

`0` indicates success, `1` indicates compilation or I/O failure, and `2` indicates invalid CLI arguments.

## wave-emu

Execute a `.wbin` program on the CPU.

### Usage

```text
wave-emu <program.wbin> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `program.wbin` | Binary file |

### Options

| Option | Default | Description |
|---|---|---|
| `--grid X,Y,Z` | `1,1,1` | Workgroup counts in each dimension |
| `--workgroup X,Y,Z` | `32,1,1` | Workgroup dimensions when kernel metadata does not supply them |
| `--registers N` | `32` | Registers per thread when kernel metadata does not supply a count |
| `--local-memory N` | `16384` | Local-memory bytes when kernel metadata does not supply a size |
| `--device-memory N` | `1048576` | Device-memory bytes |
| `--wave-width N` | `32` | Lanes per wave |
| `--max-instructions N` | `10000000` | Instruction budget; `0` removes the limit |
| `--trace` | Off | Print executed instructions |
| `--stats` | Off | Print operation, memory, workgroup, wave, and barrier counters |
| `--arg OFFSET:FILE` | None | Load a file into device memory at the byte offset |
| `--set-reg REG:VALUE` | None | Set a register in every thread |
| `--fill-iota OFFSET:TYPE:COUNT:SCALE` | None | Fill memory with `0`, `SCALE`, `2*SCALE`, and subsequent values |
| `--fill-zero OFFSET:TYPE:COUNT` | None | Fill memory with zeros |
| `--dump-regs` | Off | Print registers after execution |
| `--dump-memory START:END` | None | Print the specified byte range as hexadecimal |
| `--dump-u32 OFFSET:COUNT` | None | Print memory as `u32` values |
| `--dump-f32 OFFSET:COUNT` | None | Print memory as `f32` values |
| `--dump-f16 OFFSET:COUNT` | None | Print memory as `f16` values |
| `--dump-bf16 OFFSET:COUNT` | None | Print memory as `bf16` values |

### Memory initialization

`--arg` loads raw file bytes at the specified byte offset. Repeat it to load several regions:

```bash
wave-emu kernel.wbin --grid 1,1,1 --workgroup 64,1,1 \
  --arg 0:input.bin --arg 256:output.bin
```

### Examples

```bash
wave-emu program.wbin --grid 1,1,1 --workgroup 64,1,1
wave-emu program.wbin --grid 4,1,1 --workgroup 256,1,1
wave-emu volume.wbin --grid 8,8,8 --workgroup 4,4,4
wave-emu debug.wbin --wave-width 64 --trace
wave-emu matmul.wbin --grid 16,16,1 --workgroup 16,16,1 \
  --device-memory 16777216 --local-memory 49152 --stats
wave-emu kernel.wbin --registers 16 --dump-regs
```

Kernel metadata overrides nonzero register and local-memory declarations and declared workgroup dimensions.

### Exit codes

`0` indicates success, `1` indicates execution or I/O failure, and `2` indicates invalid CLI arguments. Exceeding the instruction budget is an execution failure.

### Trace output format

Trace records identify the workgroup, wave, program counter, and instruction:

```text
wg(0,0,0) wave[0] pc=0x0000: mov_imm r0, 3
```

### Statistics output

`--stats` reports executed instructions by category, device and local memory loads and stores with byte counts, barriers, divergent branches, workgroups, and waves. It does not estimate GPU cycles.
