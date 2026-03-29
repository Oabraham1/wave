// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the WAVE Rust SDK.

use wave_sdk::{array, device, kernel};

#[test]
fn test_device_detection() {
    let dev = device::detect().unwrap();
    assert!(!dev.name.is_empty());
}

#[test]
fn test_array_f32_roundtrip() {
    let buf = array::from_f32(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(buf.count, 4);
    assert_eq!(buf.to_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_array_zeros() {
    let buf = array::zeros_f32(8);
    assert_eq!(buf.count, 8);
    assert_eq!(buf.to_f32().unwrap(), vec![0.0; 8]);
}

#[test]
fn test_compile_kernel() {
    let source = r#"
@kernel
def vector_add(a: Buffer[f32], b: Buffer[f32], out: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"#;
    let result = kernel::compile(source, kernel::Language::Python);
    assert!(result.is_ok());
}

#[test]
fn test_vector_add_emulator() {
    let source = r#"
@kernel
def vector_add(a: Buffer[f32], b: Buffer[f32], out: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"#;
    let dev = device::Device {
        vendor: device::GpuVendor::Emulator,
        name: "Test Emulator".into(),
    };

    let mut kern = kernel::compile(source, kernel::Language::Python).unwrap();

    let mut a = array::from_f32(&[1.0, 2.0, 3.0, 4.0]);
    let mut b = array::from_f32(&[5.0, 6.0, 7.0, 8.0]);
    let mut out = array::zeros_f32(4);

    kern.launch(
        &dev,
        &mut [&mut a, &mut b, &mut out],
        &[4],
        [1, 1, 1],
        [4, 1, 1],
    )
    .unwrap();

    assert_eq!(out.to_f32().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
}
