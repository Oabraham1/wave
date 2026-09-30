---
title: "Rust API reference"
description: "API reference for the wave_gpu Rust crate."
---

The `wave-gpu` crate is imported as `wave_gpu`. It provides host buffers, device detection, kernel compilation, and dispatch.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
wave-gpu = "0.1"
```

## `wave_gpu::array`

Host buffer creation and decoding.

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
pub struct DeviceBuffer {
    pub data: Vec<u8>,
    pub count: usize,
    pub element_type: ElementType,
}
```

A host buffer passed to the runtime for dispatch.

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Vec<u8>` | Little-endian element bytes. |
| `count` | `usize` | Number of elements. |
| `element_type` | `ElementType` | Element format. |

Methods:

##### `to_f32(&self) -> Result<Vec<f32>, RuntimeError>`

Decode the buffer as `f32` values. Returns a memory error for a different element type or invalid byte length.

```rust
let buf = wave_gpu::array::from_f32(&[1.0, 2.0, 3.0]);
let host: Vec<f32> = buf.to_f32()?;
assert_eq!(host, vec![1.0, 2.0, 3.0]);
```

##### `to_u32(&self) -> Result<Vec<u32>, RuntimeError>`

Decode the buffer as `u32` values. Returns a memory error for a different element type or invalid byte length.

```rust
let buf = wave_gpu::array::from_u32(&[10, 20, 30]);
let host: Vec<u32> = buf.to_u32()?;
assert_eq!(host, vec![10, 20, 30]);
```

### Free functions

#### `from_f32(data: &[f32]) -> DeviceBuffer`

Create a host buffer from a host `f32` slice. Copies the data into host storage.

```rust
let buf = wave_gpu::array::from_f32(&[1.0, 2.0, 3.0, 4.0]);
assert_eq!(buf.count, 4);
```

#### `zeros_f32(n: usize) -> DeviceBuffer`

Create a host buffer of `n` zero-valued `f32` elements.

```rust
let buf = wave_gpu::array::zeros_f32(1024);
assert_eq!(buf.count, 1024);
```

#### `from_u32(data: &[u32]) -> DeviceBuffer`

Create a host buffer from a host `u32` slice. Copies the data into host storage.

```rust
let buf = wave_gpu::array::from_u32(&[0, 1, 2, 3]);
assert_eq!(buf.count, 4);
```

#### `zeros_u32(n: usize) -> DeviceBuffer`

Create a host buffer of `n` zero-valued `u32` elements.

```rust
let buf = wave_gpu::array::zeros_u32(256);
assert_eq!(buf.count, 256);
```

---

## `wave_gpu::device`

GPU device detection and information.

### Types

#### `GpuVendor`

```rust
pub enum GpuVendor {
    Apple,
    Nvidia,
    Amd,
    Intel,
    Emulator,
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

### Free functions

#### `detect() -> Result<Device, RuntimeError>`

Detect an available device. Uses the emulator when no GPU is found.

```rust
use wave_gpu::device;

if let Ok(dev) = device::detect() {
    println!("Found GPU: {} ({:?})", dev.name, dev.vendor);
} else {
    eprintln!("No GPU detected");
}
```

---

## `wave_gpu::kernel`

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

### Free functions

#### `compile(source: &str, lang: Language) -> Result<CompiledKernel, RuntimeError>`

Compile a kernel from source code. Returns a `CompiledKernel` on success, or a `RuntimeError` describing the compilation failure.

```rust
use wave_gpu::kernel::{compile, Language};

let source = r#"
@wave_gpu.kernel
def add(a: f32[:], b: f32[:], out: f32[:]):
    tid = thread_id()
    out[tid] = a[tid] + b[tid]
"#;

let kernel = compile(source, Language::Python).expect("compilation failed");
```

### `CompiledKernel` methods

#### `launch(&mut self, device: &Device, buffers: &mut [&mut DeviceBuffer], scalars: &[u32], grid: [u32; 3], workgroup: [u32; 3]) -> Result<(), RuntimeError>`

Dispatch the kernel on the given device.

```rust
use wave_gpu::{array, device, kernel};

let dev = device::detect().expect("no GPU");
let mut a = array::from_f32(&[1.0, 2.0, 3.0, 4.0]);
let mut b = array::from_f32(&[5.0, 6.0, 7.0, 8.0]);
let mut out = array::zeros_f32(4);

let mut k = kernel::compile(src, kernel::Language::Python).unwrap();

k.launch(
    &dev,
    &mut [&mut a, &mut b, &mut out],
    &[],              // no scalar arguments
    [1, 1, 1],        // workgroup counts
    [4, 1, 1],        // workgroup dimensions
).expect("kernel launch failed");

let result = out.to_f32()?;
assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `&Device` | Target GPU device. |
| `buffers` | `&mut [&mut DeviceBuffer]` | Device buffers bound as kernel arguments, in declaration order. |
| `scalars` | `&[u32]` | Scalar values passed as kernel arguments; pushed as 32-bit constants. |
| `grid` | `[u32; 3]` | Workgroup counts `[x, y, z]`. |
| `workgroup` | `[u32; 3]` | Workgroup dimensions `[x, y, z]`. |

Returns: `Ok(())` on success. `Err(RuntimeError)` if translation, buffer preparation, compilation, or execution fails.
