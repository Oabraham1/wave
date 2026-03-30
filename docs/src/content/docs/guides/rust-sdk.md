---
title: Rust SDK
description: Getting started with the WAVE Rust SDK (wave-sdk) for portable GPU programming.
---

The `wave-sdk` crate provides a safe Rust interface for compiling and launching GPU kernels through the WAVE toolchain. This guide covers device detection, buffer management, kernel compilation, and error handling.

## Installation

Add the crate to your project:

```bash
cargo add wave-sdk
```

Or add it manually in `Cargo.toml`:

```toml
[dependencies]
wave-sdk = "0.1"
```

For local development against a checkout of the WAVE repo you can use a path dependency:

```toml
[dependencies]
wave-sdk = { path = "../wave-sdk" }
```

## Device Detection

Call `wave_sdk::device::detect()` to probe for a GPU. It returns a `Result<Device>`:

```rust
use wave_sdk::device;

fn main() -> Result<(), wave_sdk::RuntimeError> {
    let dev = device::detect()?;
    println!("vendor: {:?}, name: {}", dev.vendor, dev.name);
    Ok(())
}
```

The `vendor` field is a `GpuVendor` enum with the following variants:

| Variant | Description |
|---|---|
| `Apple` | Apple integrated GPU (Metal) |
| `Nvidia` | NVIDIA discrete GPU |
| `Amd` | AMD discrete or integrated GPU |
| `Intel` | Intel integrated GPU |
| `Emulator` | Software emulator backend |

## Creating Buffers

Device buffers are created through factory functions in `wave_sdk::array`. Each function returns a `DeviceBuffer`:

```rust
use wave_sdk::array;

// From a slice of f32 values
let a = array::from_f32(&[1.0, 2.0, 3.0, 4.0])?;

// Zero-initialized buffer of 1024 f32 elements
let out = array::zeros_f32(1024)?;

// Unsigned integer buffers
let indices = array::from_u32(&[0, 1, 2, 3])?;
let counts  = array::zeros_u32(256)?;
```

`DeviceBuffer` exposes:

| Member | Description |
|---|---|
| `count` | Number of elements in the buffer |
| `to_f32()` | Copy contents to a `Vec<f32>` |
| `to_u32()` | Copy contents to a `Vec<u32>` |

## Writing and Compiling Kernels

WAVE compiles kernel source written in any of its supported front-end languages. Pass the source string and a `Language` variant to `wave_sdk::kernel::compile`:

```rust
use wave_sdk::kernel::{compile, Language};

let source = r#"
import wave_gpu

@wave_gpu.kernel
def vector_add(a, b, out, n):
    tid = wave_gpu.thread_id()
    if tid < n:
        out[tid] = a[tid] + b[tid]
"#;

let kernel = compile(source, Language::Python)?;
```

Available `Language` variants: `Python`, `Rust`, `Cpp`, `Typescript`.

## Launching Kernels

`CompiledKernel::launch` dispatches the kernel on a device. You provide the device handle, a slice of buffer references, a slice of scalar values, and the grid / workgroup dimensions:

```rust
let n: u32 = 1024;
let a   = array::from_f32(&vec![1.0; n as usize])?;
let b   = array::from_f32(&vec![2.0; n as usize])?;
let out = array::zeros_f32(n as usize)?;

let dev = device::detect()?;

kernel.launch(
    &dev,
    &[&a, &b, &out],       // buffers
    &[n.into()],            // scalars
    [n / 256, 1, 1],        // grid dimensions
    [256, 1, 1],            // workgroup dimensions
)?;
```

Both `grid` and `workgroup` are `[u32; 3]` arrays representing `(x, y, z)` dimensions.

## Reading Results

After launch completes, copy data back to the host:

```rust
let result: Vec<f32> = out.to_f32()?;
assert_eq!(result[0], 3.0);
```

## Error Handling

Every fallible operation in `wave-sdk` returns `Result<T, RuntimeError>`. `RuntimeError` covers device detection failures, compilation errors, and launch failures. Use the standard `?` operator or pattern-match:

```rust
match device::detect() {
    Ok(dev) => println!("Found {}", dev.name),
    Err(e)  => eprintln!("No device: {e}"),
}

match kernel.launch(&dev, &[&a, &b, &out], &[n.into()], [4, 1, 1], [256, 1, 1]) {
    Ok(()) => {},
    Err(e) => eprintln!("Launch failed: {e}"),
}
```

## Next Steps

See the full [Rust API Reference](/reference/rust-api) for detailed type documentation and additional utilities.
