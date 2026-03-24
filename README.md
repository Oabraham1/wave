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
| `wave-metal` | Apple Metal backend. `.wbin` → `.metal` MSL source. |
| `wave-ptx` | NVIDIA PTX backend. `.wbin` → `.ptx` assembly. |
| `wave-hip` | AMD HIP backend. `.wbin` → `.hip` C++ source. |
| `wave-sycl` | Intel SYCL backend. `.wbin` → `.cpp` SYCL source. |

## Build

```
cd wave-asm && cargo build --release
cd wave-decode && cargo build --release
cd wave-dis && cargo build --release
cd wave-emu && cargo build --release
cd wave-metal && cargo build --release
cd wave-ptx && cargo build --release
cd wave-hip && cargo build --release
cd wave-sycl && cargo build --release
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

## Backends

The same WAVE binary runs on any GPU through vendor-specific backends:

| Backend | Target | GPU | Verified |
|---------|--------|-----|----------|
| wave-metal | Apple Metal | M1, M2, M3, M4 | Yes |
| wave-ptx | NVIDIA PTX | T4, A100, H100 | Yes |
| wave-hip | AMD HIP/ROCm | MI250, MI300X, RX 7000 | Pending |
| wave-sycl | Intel SYCL/oneAPI | Arc, Max, Flex | Pending |

## Verify

```
cd tests/spec-verification && ./run_all.sh
```

102 tests verify every claim in the WAVE v0.1 spec against the reference toolchain.

## Tests

| Crate | Tests |
|-------|-------|
| wave-decode | 16 |
| wave-asm | 78 |
| wave-dis | 22 |
| wave-emu | 75 |
| wave-metal | 77 |
| wave-ptx | 76 |
| wave-hip | 53 |
| wave-sycl | 50 |
| spec-verification | 102 |
| **Total** | **549** |

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](LICENSE) for terms.
