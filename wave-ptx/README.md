# wave-ptx

NVIDIA PTX backend for the WAVE ISA. Translates `.wbin` binaries to PTX assembly (`.ptx`).

## Build

```
cd wave-ptx && cargo build --release
```

## Usage

```
wave-ptx program.wbin -o program.ptx
wave-ptx program.wbin --sm 75 -o program.ptx
```

Default SM target is 75 (Turing / T4). Override with `--sm`.

## Pipeline

```bash
wave-asm vector_add.wave -o vector_add.wbin
wave-ptx vector_add.wbin -o vector_add.ptx
python examples/run_on_gpu.py vector_add.ptx
```

Requires `pycuda` and an NVIDIA GPU.

## Instruction Mapping

| WAVE | PTX |
|------|-----|
| Registers | `%r<N>` (b32), `%f<N>` (f32), `%rd<N>` (b64), `%p<4>` (pred) |
| Float bitcast | `mov.b32 %f, %r` / `mov.b32 %r, %f` between register sets |
| Local memory | `.shared .align 4 .b8 _shared_mem[N]` |
| Device memory | `ld.global.u32` / `st.global.u32` via 64-bit address (`cvt.u64.u32` + `add.u64`) |
| Barrier | `bar.sync 0` |
| Fence | `membar.cta` / `membar.gl` / `membar.sys` |
| Shuffle | `shfl.sync.idx.b32` / `.up` / `.down` / `.bfly` with `0xFFFFFFFF` mask |
| Reduce | 5-step butterfly tree via `shfl.sync.bfly.b32` |
| Prefix sum | 5-step `shfl.sync.up.b32` with predicated accumulation |
| Ballot | `vote.sync.ballot.b32` |
| Vote | `vote.sync.any.pred` / `vote.sync.all.pred` |
| Atomics | `atom.global.add.u32` / `atom.shared.add.u32` (no `atom.sub` — uses negation + add) |
| Control flow | Predicated branches (`@%p bra $label`) with generated labels |
| Predication | PTX instruction prefix: `@%p1 add.s32 ...` / `@!%p0 mov.b32 ...` |
| Special registers | PTX built-ins: `%tid.x`, `%ctaid.x`, `%ntid.x`, `%laneid`, etc. |

## Design Notes

PTX is assembly, not C++. This makes it fundamentally different from the Metal, HIP, and SYCL backends:

- **Control flow lowering.** Structured `if`/`else`/`endif` and `loop`/`break`/`endloop` are lowered to predicated branches with generated labels (`$L_else_0:`, `$L_loop_1:`). A label counter and stack track nesting.
- **Float bitcasting.** WAVE registers are untyped 32-bit. PTX registers are typed. Every float operation requires `mov.b32` to copy between `%r` (b32) and `%f` (f32) register sets before and after the operation.
- **64-bit addressing.** Device memory requires 64-bit addresses. The 32-bit WAVE byte offset is zero-extended via `cvt.u64.u32` and added to the base pointer in `%rd0`.
- **No atomic sub.** PTX lacks `atom.sub`. The backend negates the value and uses `atom.add`.

## Tests

76 tests across 5 test files:
- 47 arithmetic (integer, float, bitwise, comparison, conversion, predicates, kernel structure)
- 9 memory (global/shared load/store, atomics, CAS, fences)
- 4 control flow (if/endif, if/else/endif, loop/break, loop/continue)
- 12 wave ops (shuffle modes, ballot, vote, reduce add/min/max, prefix sum)
- 4 programs (vector_add, reduction with barrier, empty kernel, invalid WBIN)

```
cargo test
```

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](../LICENSE) for terms.
