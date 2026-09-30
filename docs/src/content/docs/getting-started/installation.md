---
title: "Installation"
description: "Build the WAVE toolchain and install the Python, Rust, C++, or TypeScript SDK."
---

These instructions build WAVE from a checkout. Each crate builds independently.

## Toolchain

Install Git and a stable Rust toolchain through [rustup](https://rustup.rs/), then build the compiler, assembler, emulator, runtime, and backends:

```sh
git clone https://github.com/Oabraham1/wave.git
cd wave
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime; do
  cargo build --release --manifest-path "$crate/Cargo.toml"
done
```

Each crate writes its output to its own `target/release/` directory. The runtime build also produces the shared library used by the Python SDK. Keep the checkout available when using the SDKs from source.

To put the command-line tools on your `PATH`:

```sh
cargo install --path wave-compiler
cargo install --path wave-asm
cargo install --path wave-dis
cargo install --path wave-emu
```

## Python

From the repository root:

```sh
python3 -m pip install -e ./sdk/python
```

Check the installed version and device detection:

```python
import wave_gpu

print(wave_gpu.__version__)
print(wave_gpu.device())
```

Continue to [Quick start](/getting-started/quickstart/) for a complete emulator example.

## Rust

Add the SDK to your application, using the path to your checkout:

```sh
cargo add wave-gpu --path /path/to/wave/sdk/rust
```

The crate is imported as `wave_gpu`:

```rust
use wave_gpu::device;

fn main() -> Result<(), wave_gpu::RuntimeError> {
    let dev = device::detect()?;
    println!("{}", dev.name);
    Ok(())
}
```

## C++

The C++ SDK requires C++17 and CMake 3.16 or later. Build it from the repository root:

```sh
cmake -S sdk/cpp -B sdk/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build sdk/cpp/build
ctest --test-dir sdk/cpp/build --output-on-failure
```

To include the SDK in a CMake project:

```cmake
add_subdirectory(/path/to/wave/sdk/cpp wave-sdk)
target_link_libraries(your_target PRIVATE wave)
```

Headers are available through `#include <wave/wave.h>`. See the [C++ guide](/guides/cpp-sdk/) for buffer ownership and launch arguments.

## TypeScript

The TypeScript SDK requires Node.js 18 or later. Build it in the checkout:

```sh
cd sdk/js
npm install
npm run build
```

From your application directory, install the local package:

```sh
npm install /path/to/wave/sdk/js
```

Detect the device:

```typescript
import { device } from "wave-gpu";

async function main() {
  console.log(await device());
}

main();
```

## GPU toolchains

Vendor compilation and dispatch require the corresponding toolchain. Emulator execution does not require a vendor SDK.

| Backend | Toolchain |
|---|---|
| `wave-metal` | Xcode command-line tools and Metal |
| `wave-ptx` | NVIDIA CUDA toolkit |
| `wave-hip` | AMD ROCm and `hipcc` |
| `wave-sycl` | Intel oneAPI and `dpcpp` |

## Tests

Run from the repository root:

```sh
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime sdk/rust tests/spec-verification; do
  cargo test --manifest-path "$crate/Cargo.toml"
done
bash tests/ci/run_emulator_tests.sh
bash tests/ci/run_backend_tests.sh
```
