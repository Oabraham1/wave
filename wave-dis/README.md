# wave-dis

Disassembler for the WAVE ISA.

Reads WAVE binaries (`.wbin`) and produces human-readable assembly text. Inverse of `wave-asm`.

Spec: https://ojima.me/spec.html

## Build

```
cargo build --release
```

## Usage

```
wave-dis program.wbin
wave-dis program.wbin -o disassembled.wave
wave-dis program.wbin --offsets --raw
```

## Options

| Flag | Description |
|------|-------------|
| `-o <file>` | Output file (default: stdout) |
| `--offsets` | Show byte offsets |
| `--raw` | Show raw hex encoding |
| `--no-directives` | Omit `.kernel`/`.registers`/`.end` directives |

## Round-Trip

```bash
wave-asm input.wave -o tmp.wbin
wave-dis tmp.wbin -o roundtrip.wave
wave-asm roundtrip.wave -o roundtrip.wbin
diff tmp.wbin roundtrip.wbin  # must be identical
```

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](../LICENSE) for terms.
