---
title: "Rust SDK"
description: "Device detection, buffers, compilation, and dispatch with wave-gpu."
---

The `wave-gpu` crate provides device detection, host buffers, kernel compilation, and dispatch through the WAVE runtime.

## Installation

Add the crate to your project:

```bash
cargo add wave-gpu
```

Or add it manually in `Cargo.toml`:

```toml
[dependencies]
wave-gpu = "0.1"
```

For local development against a checkout of the WAVE repo you can use a path dependency:

```toml
[dependencies]
wave-gpu = { path = "../wave/sdk/rust" }
```

## Device detection

Call `wave_gpu::device::detect()` to probe for a GPU. It returns a `Result<Device>`:

```rust
use wave_gpu::device;

fn main() -> Result<(), wave_gpu::RuntimeError> {
    let dev = device::detect()?;
    println!("vendor: {:?}, name: {}", dev.vendor, dev.name);
    Ok(())
}
```

`GpuVendor` has these variants:

| Variant | Description |
|---|---|
| `Apple` | Apple integrated GPU; Metal |
| `Nvidia` | NVIDIA discrete GPU |
| `Amd` | AMD discrete or integrated GPU |
| `Intel` | Intel integrated GPU |
| `Emulator` | Software emulator backend |

## Creating buffers

Device buffers are created through factory functions in `wave_gpu::array`. Each function returns a host-side `DeviceBuffer`:

```rust
use wave_gpu::array;

// From a slice of f32 values
let a = array::from_f32(&[1.0, 2.0, 3.0, 4.0]);

// Zero-initialized buffer of 1024 f32 elements
let out = array::zeros_f32(1024);

// Unsigned integer buffers
let indices = array::from_u32(&[0, 1, 2, 3]);
let counts  = array::zeros_u32(256);
```

`DeviceBuffer` exposes:

| Member | Description |
|---|---|
| `count` | Number of elements in the buffer |
| `to_f32()` | Copy contents to a `Vec<f32>` |
| `to_u32()` | Copy contents to a `Vec<u32>` |

## Writing and compiling kernels

WAVE compiles kernel source written in any of its supported front-end languages. Pass the source string and a `Language` variant to `wave_gpu::kernel::compile`:

```rust
use wave_gpu::kernel::{compile, Language};

let source = r#"
import wave_gpu

@wave_gpu.kernel
def vector_add(a: f32[:], b: f32[:], out: f32[:], n: u32):
    tid = thread_id()
    if tid < n:
        out[tid] = a[tid] + b[tid]
"#;

let mut kernel = compile(source, Language::Python)?;
```

Available `Language` variants: `Python`, `Rust`, `Cpp`, `TypeScript`.

## Launching kernels

`CompiledKernel::launch` dispatches the kernel on a device. You provide the device handle, mutable buffer references, scalar values, and grid and workgroup dimensions:

```rust
let n: u32 = 1024;
let mut a = array::from_f32(&vec![1.0; n as usize]);
let mut b = array::from_f32(&vec![2.0; n as usize]);
let mut out = array::zeros_f32(n as usize);

let dev = device::detect()?;

kernel.launch(
    &dev,
    &mut [&mut a, &mut b, &mut out],
    &[n],
    [n / 256, 1, 1],
    [256, 1, 1],
)?;
```

Both `grid` and `workgroup` are `[u32; 3]` arrays representing `[x, y, z]` dimensions.

## Reading results

After launch, read the updated buffer:

```rust
let result: Vec<f32> = out.to_f32()?;
assert_eq!(result[0], 3.0);
```

## Error handling

Every fallible operation in `wave-gpu` returns `Result<T, RuntimeError>`. `RuntimeError` covers device detection failures, compilation errors, and launch failures. Use the standard `?` operator or pattern-match:

```rust
match device::detect() {
    Ok(dev) => println!("Found {}", dev.name),
    Err(e)  => eprintln!("No device: {e}"),
}

match kernel.launch(&dev, &mut [&mut a, &mut b, &mut out], &[n], [4, 1, 1], [256, 1, 1]) {
    Ok(()) => {},
    Err(e) => eprintln!("Launch failed: {e}"),
}
```

## Next steps

See the [Rust API reference](/reference/rust-api) for types and signatures.
