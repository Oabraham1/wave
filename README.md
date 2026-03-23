# wave

Toolchain for the WAVE instruction set architecture — a vendor-neutral ISA for general-purpose GPU computation.

Spec: https://ojima.me/spec.html

## Tools

| Tool | Description |
|------|-------------|
| `wave-asm` | Assembler. `.wave` text → `.wbin` binary. |
| `wave-decode` | Shared decoder library. `.wbin` binary → structured types. |
| `wave-dis` | Disassembler. `.wbin` binary → `.wave` text. |
| `wave-emu` | Reference emulator. Executes `.wbin` on the CPU. |

## Build

```
cd wave-asm && cargo build --release
cd wave-decode && cargo build --release
cd wave-dis && cargo build --release
cd wave-emu && cargo build --release
```

## Example

```bash
# Write a kernel
cat > add.wave << 'EOF'
.kernel vec_add
.registers 8
.workgroup_size 256, 1, 1

    mov r3, sr_workgroup_id_x
    mov r4, sr_workgroup_size_x
    imul r3, r3, r4
    mov r4, sr_thread_id_x
    iadd r3, r3, r4

    mov_imm r4, 4
    imul r4, r3, r4

    iadd r5, r0, r4
    device_load_u32 r5, r5
    iadd r6, r1, r4
    device_load_u32 r6, r6
    wait

    fadd r7, r5, r6

    iadd r5, r2, r4
    device_store_u32 r5, r7

    halt
.end
EOF

# Assemble
wave-asm add.wave -o add.wbin

# Inspect
wave-dis add.wbin

# Run
wave-emu add.wbin --grid 4,1,1 --workgroup 256,1,1 --stats
```

## Verify

```
cd tests/spec-verification && ./run_all.sh
```

102 tests verify every claim in the WAVE v0.1 spec against the reference toolchain.

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](LICENSE) for terms.
