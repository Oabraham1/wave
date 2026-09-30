// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Latency of assembly, decoding, compilation, backend emission, shuffles, and reductions.

use std::hint::black_box;
use std::time::Instant;
use wave_compiler::Language;
use wave_decode::WbinFile;
use wave_emu::wave::{Wave, WaveConfig};
use wave_runtime::{backend, cache, compiler, device::GpuVendor, memory::DeviceBuffer, multi};

const SOURCE: &str = "@kernel\ndef add(a: Buffer[f32], b: Buffer[f32], out: Buffer[f32], n: u32):\n    gid = thread_id()\n    if gid < n:\n        out[gid] = a[gid] + b[gid]\n";

fn measure<T>(workload: &str, units: u32, repetitions: u32, mut run: impl FnMut() -> T) {
    black_box(run());
    let mut times = Vec::with_capacity(7);
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..repetitions {
            black_box(run());
        }
        times.push(start.elapsed() / repetitions);
    }
    times.sort_unstable();
    let median = times[3];
    println!(
        "{workload},{units},{},{:.2}",
        median.as_nanos(),
        f64::from(units) / median.as_secs_f64()
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("workload,units,median_ns,units_per_second");
    let assembly = format!(
        ".kernel arithmetic\n.registers 8\n{}halt\n.end\n",
        "iadd r0, r1, r2\nfma r3, r4, r5, r6\n".repeat(512)
    );
    let binary = wave_asm::assemble(&assembly, "arithmetic.wave")?.binary;
    let wbin = WbinFile::parse(&binary)?;
    assert_eq!(wave_decode::decode_all(wbin.code())?.len(), 1025);
    measure("assemble_instructions", 1025, 100, || {
        wave_asm::assemble(black_box(&assembly), "arithmetic.wave").unwrap()
    });
    measure("decode_instructions", 1025, 1000, || {
        wave_decode::decode_all(black_box(wbin.code())).unwrap()
    });
    measure("disassemble_instructions", 1025, 100, || {
        wave_dis::disassemble_code(
            black_box(wbin.code()),
            &wave_dis::DisassemblyOptions::default(),
        )
        .unwrap()
    });
    measure("compile_vector_add", 1, 100, || {
        compiler::compile_kernel(black_box(SOURCE), Language::Python).unwrap()
    });
    let mut stores = String::from(
        "@kernel\ndef stores(a: Buffer[f32], out: Buffer[f32]):\n    gid = thread_id()\n",
    );
    for index in 0..128 {
        use std::fmt::Write as _;
        writeln!(stores, "    out[gid] = a[gid] * 1.25 + {index}.0").unwrap();
    }
    measure("compile_float_stores", 128, 10, || {
        compiler::compile_kernel(black_box(&stores), Language::Python).unwrap()
    });
    for vendor in [
        GpuVendor::Apple,
        GpuVendor::Nvidia,
        GpuVendor::Amd,
        GpuVendor::Intel,
    ] {
        measure(&format!("emit_{vendor:?}_instructions"), 1025, 100, || {
            backend::translate_to_vendor(black_box(&binary), vendor).unwrap()
        });
    }
    cache::compile_cached(SOURCE, Language::Python)?;
    measure("cached_compile", 1, 10_000, || {
        cache::compile_cached(black_box(SOURCE), Language::Python).unwrap()
    });
    measure_execution(wbin.code());
    measure_reductions()?;
    Ok(())
}

fn measure_execution(code: &[u8]) {
    for width in [4, 32, 64] {
        let mut wave = Wave::new(WaveConfig {
            wave_width: width,
            register_count: 4,
            wave_id: 0,
            workgroup_id: [0; 3],
            workgroup_size: [width, 1, 1],
            grid_size: [1; 3],
            base_thread_index: 0,
            total_threads_in_workgroup: width,
            num_waves: 1,
        });
        for (lane, thread) in wave.threads.iter_mut().enumerate() {
            thread.write_register(0, u32::try_from(lane).unwrap());
            thread.write_register(1, 1);
        }
        wave_emu::shuffle::wave_shuffle_xor(&mut wave, 2, 0, 1);
        assert_eq!(wave.threads[0].read_register(2), 1);
        let config = wave_emu::EmulatorConfig {
            wave_width: width,
            workgroup_dim: [width, 1, 1],
            register_count: 8,
            device_memory_size: 16,
            local_memory_size: 16,
            ..wave_emu::EmulatorConfig::default()
        };
        let mut execution_memory = wave_emu::memory::DeviceMemory::new(16);
        measure("execute_arithmetic_instructions", 1025 * width, 100, || {
            wave_emu::core::Core::new(&config, black_box(code), &mut execution_memory, [0; 3])
                .run()
                .unwrap()
        });
        let mut executor = wave_emu::executor::Executor::new(code, false, [0; 3]);
        let mut stats = wave_emu::stats::ExecutionStats::new();
        let mut local = wave_emu::memory::LocalMemory::new(16);
        let mut device = wave_emu::memory::DeviceMemory::new(16);
        measure("execute_iadd_lanes", width, 100_000, || {
            wave.pc = 0;
            executor
                .step(black_box(&mut wave), &mut local, &mut device, &mut stats)
                .unwrap()
        });
        measure("shuffle_xor_lanes", width, 100_000, || {
            wave_emu::shuffle::wave_shuffle_xor(black_box(&mut wave), 2, 0, 1);
        });
    }
}

fn measure_reductions() -> Result<(), Box<dyn std::error::Error>> {
    for count in [256_u32, 262_144] {
        let mut buffers = vec![DeviceBuffer::from_f32(&vec![1.0; count as usize]); 4];
        multi::allreduce_average(&mut buffers)?;
        assert!(buffers[0]
            .to_f32()?
            .iter()
            .all(|&value| value.to_bits() == 1.0_f32.to_bits()));
        measure("allreduce_f32_elements", count * 4, 100, || {
            multi::allreduce_average(black_box(&mut buffers)).unwrap();
        });
    }
    Ok(())
}
