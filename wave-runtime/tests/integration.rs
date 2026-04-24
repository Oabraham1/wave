// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the WAVE runtime.

use wave_runtime::*;

#[test]
fn test_full_pipeline_emulator() {
    let source = r#"
@kernel
def vector_add(a: Buffer[f32], b: Buffer[f32], out: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"#;

    let wbin = compile_kernel(source, Language::Python).unwrap();
    assert_eq!(&wbin[0..4], b"WAVE");

    let mut a = DeviceBuffer::from_f32(&[1.0, 2.0, 3.0, 4.0]);
    let mut b = DeviceBuffer::from_f32(&[5.0, 6.0, 7.0, 8.0]);
    let mut out = DeviceBuffer::zeros_f32(4);

    launch_kernel(
        "",
        &wbin,
        GpuVendor::Emulator,
        &mut [&mut a, &mut b, &mut out],
        &[4],
        [1, 1, 1],
        [4, 1, 1],
    )
    .unwrap();

    assert_eq!(out.to_f32().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_device_detection() {
    let device = detect_gpu().unwrap();
    assert!(!device.name.is_empty());
}

#[test]
fn test_backend_translation_metal() {
    let source = r#"
@kernel
def noop(n: u32):
    gid = thread_id()
"#;

    let wbin = compile_kernel(source, Language::Python).unwrap();

    if cfg!(target_os = "macos") {
        let msl = translate_to_vendor(&wbin, GpuVendor::Apple).unwrap();
        assert!(msl.contains("kernel"));
    }
}

#[test]
fn test_large_array_emulator() {
    let source = r#"
@kernel
def vector_add(a: Buffer[f32], b: Buffer[f32], out: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"#;

    let n = 256;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let expected: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();

    let wbin = compile_kernel(source, Language::Python).unwrap();

    let mut a = DeviceBuffer::from_f32(&a_data);
    let mut b = DeviceBuffer::from_f32(&b_data);
    let mut out = DeviceBuffer::zeros_f32(n);

    launch_kernel(
        "",
        &wbin,
        GpuVendor::Emulator,
        &mut [&mut a, &mut b, &mut out],
        &[n as u32],
        [8, 1, 1],
        [32, 1, 1],
    )
    .unwrap();

    assert_eq!(out.to_f32().unwrap(), expected);
}

#[test]
fn test_enumerate_devices() {
    let devices = enumerate_devices().unwrap();
    assert!(!devices.is_empty());
    let first = &devices[0];
    assert_eq!(first.id, 0);
    assert!(first.wave_width > 0);
}

#[test]
fn test_shard_and_gather_roundtrip() {
    let data = DeviceBuffer::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let devices = enumerate_devices().unwrap();
    let shards = shard_tensor(&data, &devices, 0).unwrap();
    let gathered = gather_shards(&shards).unwrap();
    assert_eq!(
        gathered.to_f32().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_allreduce_identity() {
    let mut buf = DeviceBuffer::from_f32(&[10.0, 20.0, 30.0]);
    allreduce_average(std::slice::from_mut(&mut buf)).unwrap();
    assert_eq!(buf.to_f32().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_reduce_strategy_for_system() {
    let devices = enumerate_devices().unwrap();
    let strategy = select_reduce_strategy(&devices);
    if devices.len() == 1 {
        assert_eq!(strategy, ReduceStrategy::SingleDevice);
    }
}

#[test]
fn test_replicate_buffer_matches_device_count() {
    let buf = DeviceBuffer::from_f32(&[1.0, 2.0]);
    let devices = enumerate_devices().unwrap();
    let replicas = replicate_buffer(&buf, &devices);
    assert_eq!(replicas.len(), devices.len());
    for replica in &replicas {
        assert_eq!(replica.to_f32().unwrap(), vec![1.0, 2.0]);
    }
}
