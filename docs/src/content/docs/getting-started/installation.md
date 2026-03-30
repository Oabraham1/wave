---
title: Installation
description: Install the WAVE SDK for Python, Rust, C++, or TypeScript, or build the toolchain from source.
---

WAVE provides native SDKs for four languages and a source build for the core toolchain. Pick the path that matches your project.

## Python

Requires Python 3.9 or later.

```bash
pip install wave-gpu
```

Verify the installation:

```python
import wave_gpu
print(wave_gpu.__version__)
print(wave_gpu.detect_backend())  # e.g. "metal", "ptx", "hip", or "emulator"
```

The Python package includes prebuilt binaries for the WAVE compiler and all four backends on macOS (ARM64), Linux (x86_64), and Windows (x86_64). If no supported GPU is detected at runtime, the built-in emulator is used automatically.

## Rust

Requires Rust 1.70 or later.

```bash
cargo add wave-sdk
```

Or add to your `Cargo.toml` manually:

```toml
[dependencies]
wave-sdk = "0.1"
```

Verify the installation:

```rust
use wave_sdk::device;

fn main() {
    let backend = device::detect_backend();
    println!("Backend: {:?}", backend);
}
```

The Rust SDK links against the WAVE toolchain at build time via a bundled static library. No system-level dependencies beyond a working Rust toolchain are required.

## C++

Requires a C++17-compatible compiler and CMake 3.20 or later.

### Option A: CMake FetchContent

Add to your `CMakeLists.txt`:

```cmake
include(FetchContent)
FetchContent_Declare(
  wave
  GIT_REPOSITORY https://github.com/Oabraham1/wave.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(wave)

target_link_libraries(your_target PRIVATE wave::wave-sdk)
```

### Option B: System Install

```bash
git clone https://github.com/Oabraham1/wave.git
cd wave
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

Then in your project:

```cmake
find_package(wave REQUIRED)
target_link_libraries(your_target PRIVATE wave::wave-sdk)
```

Verify the installation:

```cpp
#include <wave/wave.h>
#include <cstdio>

int main() {
    const char* backend = wave_detect_backend();
    printf("Backend: %s\n", backend);
    return 0;
}
```

## TypeScript

Requires Node.js 18 or later.

```bash
npm install wave-gpu
```

Or with other package managers:

```bash
yarn add wave-gpu
pnpm add wave-gpu
```

Verify the installation:

```typescript
import { detectBackend } from "wave-gpu";

console.log(detectBackend()); // e.g. "metal", "ptx", "hip", or "emulator"
```

The npm package ships with prebuilt native binaries for macOS (ARM64), Linux (x86_64), and Windows (x86_64) via platform-specific optional dependencies. The N-API binding loads the correct binary automatically.

## Building from Source

Building from source gives you the core WAVE toolchain: `wave-compiler`, `wave-asm`, `wave-dis`, and `wave-emu`, plus all four backends.

### Prerequisites

- **Rust toolchain** - install via [rustup](https://rustup.rs/) (stable channel, 1.70+)
- **Git**
- **CMake 3.20+** (only needed if building the C++ SDK)

### Build Steps

```bash
git clone https://github.com/Oabraham1/wave.git
cd wave
cargo build --release
```

The compiled binaries are placed in `target/release/`:

```bash
ls target/release/wave-*
# target/release/wave-compiler
# target/release/wave-asm
# target/release/wave-dis
# target/release/wave-emu
```

### Running Tests

```bash
cargo test --workspace
```

This runs unit tests, integration tests, and spec-verification tests across all crates including the emulator and every backend.

### Optional: Install to PATH

```bash
cargo install --path crates/wave-compiler
cargo install --path crates/wave-asm
cargo install --path crates/wave-dis
cargo install --path crates/wave-emu
```

This installs the tools to `~/.cargo/bin/`, which should already be on your `PATH` if you installed Rust via rustup.

### Backend-Specific Requirements

Each backend has optional vendor SDK dependencies for runtime dispatch. These are only needed if you want to compile and run on actual hardware - the emulator works without any vendor SDK.

| Backend | Vendor SDK | Environment Variable |
|---|---|---|
| `wave-metal` | Xcode Command Line Tools | Detected automatically on macOS |
| `wave-ptx` | CUDA Toolkit 11.0+ | `CUDA_PATH` |
| `wave-hip` | ROCm 5.0+ | `ROCM_PATH` |
| `wave-sycl` | oneAPI Base Toolkit 2023+ | `ONEAPI_ROOT` |

**Next:** [Quick Start](/getting-started/quickstart/) - write and run your first WAVE kernel.
