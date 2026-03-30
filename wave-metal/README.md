# wave-metal

Apple Metal backend for the WAVE ISA. Translates `.wbin` binaries to Metal Shading Language (`.metal`).

## Build

```
cd wave-metal && cargo build --release
```

## Usage

```
wave-metal program.wbin -o program.metal
```

If `-o` is omitted, MSL source is printed to stdout.

## Pipeline

```bash
wave-asm vector_add.wave -o vector_add.wbin
wave-metal vector_add.wbin -o vector_add.metal
swiftc examples/run_on_metal.swift -framework Metal -framework Foundation -o run_on_metal
./run_on_metal vector_add.metal
```

## Instruction Mapping

| WAVE | MSL |
|------|-----|
| Registers | `uint32_t` local variables |
| Float bitcast | `as_type<float>()` / `as_type<uint32_t>()` via `rf()` / `ri()` helpers |
| Local memory | `threadgroup uint8_t local_mem[N]` (static size from kernel metadata) |
| Device memory | `device uint8_t* device_mem [[buffer(0)]]` |
| Barrier | `threadgroup_barrier(mem_flags::mem_threadgroup \| mem_flags::mem_device)` |
| Fence | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Shuffle | `simd_shuffle()` / `simd_shuffle_up()` / `simd_shuffle_down()` / `simd_shuffle_xor()` |
| Broadcast | `simd_broadcast()` |
| Reduce | `simd_sum()` / `simd_min()` / `simd_max()` |
| Prefix sum | `simd_prefix_exclusive_sum()` |
| Ballot | `simd_ballot()` |
| Vote | `simd_any()` / `simd_all()` |
| Atomics | `atomic_fetch_add_explicit()` with `memory_order_relaxed` |
| Control flow | Native C++ `if`/`else`/`while` |
| Predication | Wrapped in `if (pN) { ... }` guards |
| Special registers | Kernel parameters: `thread_position_in_threadgroup`, `threadgroup_position_in_grid`, etc. |
| F16 | `half` type via `rh()` / `rhi()` helpers |
| F64 | `double` (register pair) — requires hardware support |

## Known Limitations

- No `call` instruction support. Compile error on function calls.
- F64 emits MSL `double` type. Not supported on all Apple GPUs.
- SIMD width hardcoded to 32 (`sr_wave_width` emits constant `32u`).
- Memory alignment assumed. Pointer casts require aligned byte offsets.
- Single device buffer. All device memory through `[[buffer(0)]]`.

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](../LICENSE) for terms.
