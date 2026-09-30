// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Check compare-and-swap and predicated fences on the emulator and Metal.

use std::error::Error;
use wave_runtime::{launch_kernel, DeviceBuffer, GpuVendor};

fn main() -> Result<(), Box<dyn Error>> {
    let source = r"
.kernel atomic_and_fence
.registers 16
.workgroup_size 32, 1, 1
    mov r0, sr_thread_id_x
    mov_imm r1, 16
    imul r2, r0, r1
    mov_imm r3, 5
    mov_imm r4, 10
    atomic_cas r5, r2, r3, r4, .device
    mov_imm r6, 4
    iadd r7, r2, r6
    device_store_u32 r7, r5
    mov_imm r3, 6
    mov_imm r4, 20
    atomic_cas r5, r2, r3, r4, .device
    iadd r7, r7, r6
    device_store_u32 r7, r5
    mov_imm r8, 1
    and r9, r0, r8
    mov_imm r10, 0
    icmp_eq p1, r9, r10
    @p1 fence_acquire .wave
    @p1 fence_release .workgroup
    @p1 fence_acq_rel .device
    iadd r7, r7, r6
    device_store_u32 r7, r0
    halt
.end
";
    let binary = wave_asm::assemble(source, "metal_semantics.wave")?.binary;
    let input: Vec<u32> = (0..32)
        .flat_map(|_| [5, u32::MAX, u32::MAX, u32::MAX])
        .collect();
    let expected: Vec<u32> = (0..32).flat_map(|lane| [10, 5, 10, lane]).collect();
    for vendor in [GpuVendor::Emulator, GpuVendor::Apple] {
        let mut memory = DeviceBuffer::from_u32(&input);
        launch_kernel(
            "",
            &binary,
            vendor,
            &mut [&mut memory],
            &[],
            [1, 1, 1],
            [32, 1, 1],
        )?;
        let observed = memory.to_u32()?;
        if let Some((index, (actual, reference))) = observed
            .iter()
            .zip(&expected)
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            return Err(format!(
                "atomic or fence output differs for {vendor:?} at {index}: {actual} != {reference}"
            )
            .into());
        }
        println!("{{\"workload\":\"atomic_and_predicated_fence\",\"vendor\":\"{vendor:?}\",\"correct\":true}}");
    }
    Ok(())
}
