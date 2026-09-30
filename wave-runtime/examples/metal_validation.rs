// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Check source compilation and Metal runtime launches against an independent oracle.

use std::error::Error;
use std::time::Instant;
use wave_runtime::{compile_kernel, launch_kernel, DeviceBuffer, GpuVendor, Language};

fn main() -> Result<(), Box<dyn Error>> {
    let repetitions: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "3".into())
        .parse()?;
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    let source = "@kernel\ndef add_arrays(a: Buffer[f32], b: Buffer[f32], out: Buffer[f32], n: u32):\n    gid = workgroup_id() * workgroup_size() + thread_id()\n    if gid < n:\n        out[gid] = a[gid] + b[gid]\n";
    let started = Instant::now();
    let binary = compile_kernel(source, Language::Python)?;
    let compilation_ns = started.elapsed().as_nanos();
    for n in [17_u32, 257, 4099] {
        let inputs: Vec<f32> = (0..n)
            .map(|index| f32::from(u16::try_from(index).expect("sizes fit u16")) * 0.25 - 100.0)
            .collect();
        let other: Vec<f32> = inputs.iter().map(|value| value * -0.5 + 3.0).collect();
        let expected: Vec<f32> = inputs.iter().zip(&other).map(|(a, b)| a + b).collect();
        let mut samples = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            let mut a = DeviceBuffer::from_f32(&inputs);
            let mut b = DeviceBuffer::from_f32(&other);
            let mut out = DeviceBuffer::from_f32(&vec![f32::NAN; inputs.len()]);
            let started = Instant::now();
            launch_kernel(
                "",
                &binary,
                GpuVendor::Apple,
                &mut [&mut a, &mut b, &mut out],
                &[n],
                [n.div_ceil(256), 1, 1],
                [256, 1, 1],
            )?;
            let duration = started.elapsed().as_secs_f64();
            if out.to_f32()? != expected || a.to_f32()? != inputs || b.to_f32()? != other {
                return Err(format!("Metal runtime output differs at size {n}").into());
            }
            samples.push(duration);
        }
        println!("{{\"workload\":\"runtime_vector_add\",\"size\":{n},\"correct\":true,\"source_compilation_ns\":{compilation_ns},\"launch_seconds\":{samples:?}}}");
    }
    Ok(())
}
