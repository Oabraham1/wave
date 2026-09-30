# wave-runtime

Shared runtime used by all WAVE SDKs.

Handles GPU detection, kernel compilation, backend translation, memory management, and kernel launch. All four SDKs, Python, Rust, C++, TypeScript, depend on this crate.

## Build

```bash
cargo build --release
```

## Features

- **Device detection** - Identifies Apple Metal, NVIDIA CUDA, AMD ROCm, Intel oneAPI, or falls back to the built-in emulator
- **Compilation** - Wraps wave-compiler for source-to-binary compilation
- **Backend translation** - Wraps wave-metal/ptx/hip/sycl for vendor code generation
- **Memory** - Device buffer management with f32/u32/i32 support
- **Launch** - Kernel execution via emulator or vendor toolchain subprocess

## License

Apache 2.0 - see [LICENSE](../LICENSE)

## Benchmarks

Run `cargo bench --manifest-path wave-runtime/Cargo.toml --bench toolchain` from the repository root. The benchmark reports median latency and throughput over seven rounds for assembly, decoding, disassembly, compilation, backend emission, cache hits, instruction execution, shuffles, and reductions. Setup and input checks run outside the timed regions.

## Metal validation

The Metal launcher packs buffers into a shared device allocation and initializes WAVE argument registers with their byte offsets and scalar values. It compiles a Swift host and Metal source for each launch. This subprocess path includes compilation and file transfers, so its latency is separate from resident GPU execution time.

Run the source-to-device check on an Apple GPU:

```sh
cargo run --release --manifest-path wave-runtime/Cargo.toml --example metal_validation -- 3
cargo run --release --manifest-path wave-runtime/Cargo.toml --example metal_semantics
```

The vector check covers partial and multiple workgroups with exact references and unchanged-input checks. The semantics check exercises successful and failed compare-and-swap plus predicated fences at wave, workgroup, and device scope. It does not establish general concurrent memory ordering.
