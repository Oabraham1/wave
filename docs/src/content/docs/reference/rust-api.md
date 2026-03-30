---
title: Rust API Reference
description: Complete API reference for the wave_sdk Rust crate.
---

The `wave_sdk` crate provides Rust bindings for WAVE GPU compute. It exposes device buffer management, device detection, and kernel compilation and dispatch.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
wave_sdk = "0.1"
```

## `wave_sdk::array`

Buffer creation and device memory management.

### Types

#### `ElementType`

```rust
pub enum ElementType {
    F16,
    F32,
    F64,
    I32,
    U32,
}
```

Represents the element type of a device buffer.

#### `DeviceBuffer`

```rust
pub struct DeviceBuffer { /* opaque */ }
```

A handle to a device-resident buffer. Created via the free functions in this module.

**Fields (read-only):**

| Field | Type | Description |
|-------|------|-------------|
| `count` | `usize` | Number of elements in the buffer. |

**Methods:**

##### `to_f32(&self) -> Vec<f32>`

Copy buffer contents to a host `Vec<f32>`. Panics if the buffer element type is not `F32`.

```rust
let buf = wave_sdk::array::from_f32(&[1.0, 2.0, 3.0]);
let host: Vec<f32> = buf.to_f32();
assert_eq!(host, vec![1.0, 2.0, 3.0]);
```

##### `to_u32(&self) -> Vec<u32>`

Copy buffer contents to a host `Vec<u32>`. Panics if the buffer element type is not `U32`.

```rust
let buf = wave_sdk::array::from_u32(&[10, 20, 30]);
let host: Vec<u32> = buf.to_u32();
assert_eq!(host, vec![10, 20, 30]);
```

### Free Functions

#### `from_f32(data: &[f32]) -> DeviceBuffer`

Create a device buffer from a host `f32` slice. Copies the data to device memory.

```rust
let buf = wave_sdk::array::from_f32(&[1.0, 2.0, 3.0, 4.0]);
assert_eq!(buf.count, 4);
```

#### `zeros_f32(n: usize) -> DeviceBuffer`

Create a device buffer of `n` zero-valued `f32` elements.

```rust
let buf = wave_sdk::array::zeros_f32(1024);
assert_eq!(buf.count, 1024);
```

#### `from_u32(data: &[u32]) -> DeviceBuffer`

Create a device buffer from a host `u32` slice. Copies the data to device memory.

```rust
let buf = wave_sdk::array::from_u32(&[0, 1, 2, 3]);
assert_eq!(buf.count, 4);
```

#### `zeros_u32(n: usize) -> DeviceBuffer`

Create a device buffer of `n` zero-valued `u32` elements.

```rust
let buf = wave_sdk::array::zeros_u32(256);
assert_eq!(buf.count, 256);
```

---

## `wave_sdk::device`

GPU device detection and information.

### Types

#### `GpuVendor`

```rust
pub enum GpuVendor {
    AMD,
    NVIDIA,
    Intel,
    Unknown,
}
```

#### `Device`

```rust
pub struct Device {
    pub vendor: GpuVendor,
    pub name: String,
}
```

Information about a detected GPU device.

### Free Functions

#### `detect() -> Option<Device>`

Detect the first available GPU. Returns `None` if no supported GPU is found.

```rust
use wave_sdk::device;

if let Some(dev) = device::detect() {
    println!("Found GPU: {} ({:?})", dev.name, dev.vendor);
} else {
    eprintln!("No GPU detected");
}
```

---

## `wave_sdk::kernel`

Kernel compilation and dispatch.

### Types

#### `Language`

```rust
pub enum Language {
    Python,
    Rust,
    Cpp,
    TypeScript,
}
```

Source language for kernel compilation.

#### `CompiledKernel`

```rust
pub struct CompiledKernel { /* opaque */ }
```

A compiled WAVE kernel ready for dispatch.

### Free Functions

#### `compile(source: &str, lang: Language) -> Result<CompiledKernel, String>`

Compile a kernel from source code. Returns a `CompiledKernel` on success, or an error string describing the compilation failure.

```rust
use wave_sdk::kernel::{compile, Language};

let source = r#"
@wave_gpu.kernel
def add(a: f32, b: f32, out: f32):
    tid = thread_id()
    out[tid] = a[tid] + b[tid]
"#;

let kernel = compile(source, Language::Python).expect("compilation failed");
```

### `CompiledKernel` Methods

#### `launch(&self, device: &Device, buffers: &[&DeviceBuffer], scalars: &[u32], grid: [u32; 3], workgroup: [u32; 3]) -> Result<(), String>`

Dispatch the kernel on the given device.

```rust
use wave_sdk::{array, device, kernel};

let dev = device::detect().expect("no GPU");
let a = array::from_f32(&[1.0, 2.0, 3.0, 4.0]);
let b = array::from_f32(&[5.0, 6.0, 7.0, 8.0]);
let out = array::zeros_f32(4);

let k = kernel::compile(src, kernel::Language::Python).unwrap();

k.launch(
    &dev,
    &[&a, &b, &out],
    &[],              // no scalar arguments
    [4, 1, 1],        // grid dimensions
    [4, 1, 1],        // workgroup dimensions
).expect("kernel launch failed");

let result = out.to_f32();
assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `&Device` | Target GPU device. |
| `buffers` | `&[&DeviceBuffer]` | Device buffers bound as kernel arguments, in declaration order. |
| `scalars` | `&[u32]` | Scalar values passed as kernel arguments (pushed as 32-bit constants). |
| `grid` | `[u32; 3]` | Global grid dimensions `[x, y, z]`. |
| `workgroup` | `[u32; 3]` | Workgroup dimensions `[x, y, z]`. |

**Returns:** `Ok(())` on success. `Err(String)` if dispatch fails (e.g., invalid grid dimensions, device error).
