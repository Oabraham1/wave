# wave-asm

Assembler for the WAVE ISA.

Translates WAVE assembly (`.wave`) into WAVE binary (`.wbin`).

Spec: https://ojima.me/spec.html

## Build

```
cargo build --release
```

## Usage

```
wave-asm input.wave -o output.wbin
wave-asm input.wave --dump-hex
wave-asm input.wave --dump-ast
```

## Options

| Flag | Description |
|------|-------------|
| `-o <path>` | Output file (default: input stem + `.wbin`) |
| `-v` | Verbose encoding output |
| `--dump-ast` | Print parsed AST |
| `--dump-hex` | Hex dump alongside assembly |
| `--no-symbols` | Strip symbol table |
| `-W <level>` | Warning level: `none`, `default`, `strict` |

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](LICENSE) for terms.
