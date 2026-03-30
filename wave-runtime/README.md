# wave-runtime

Shared Rust runtime used by all WAVE SDKs.

Handles GPU detection, kernel compilation, backend translation, memory management, and kernel launch. All four SDKs (Python, Rust, C++, TypeScript) depend on this crate.

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
