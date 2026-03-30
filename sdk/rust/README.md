# WAVE Rust SDK

Write GPU kernels in Rust, run on any GPU.

## Install

```toml
[dependencies]
wave-sdk = { path = "path/to/wave/sdk/rust" }
```

## Usage

```rust
use wave_sdk::{array, device, kernel};

fn main() {
    let dev = device::detect().unwrap();
    let a = array::from_f32(&[1.0, 2.0, 3.0, 4.0]);
    let b = array::from_f32(&[5.0, 6.0, 7.0, 8.0]);
    let mut out = array::zeros_f32(4);

    let kern = kernel::compile(r#"
        #[kernel]
        fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
            let gid = thread_id();
            if gid < n { out[gid] = a[gid] + b[gid]; }
        }
    "#, kernel::Language::Rust).unwrap();

    kern.launch(&dev, &[&a, &b, &mut out], &[4], [1,1,1], [256,1,1]).unwrap();
    println!("{:?}", out.to_f32());
}
```

## License

Apache 2.0 - see [LICENSE](../../LICENSE)
