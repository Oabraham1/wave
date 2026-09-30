// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Device-memory read latency and throughput, including allocation of the returned buffer.

use std::hint::black_box;
use std::time::Instant;
use wave_emu::{Emulator, EmulatorConfig, EmulatorError};

fn main() -> Result<(), EmulatorError> {
    println!("bytes,median_ns,mib_per_second");
    for (size, repetitions) in [
        (64_usize, 100_000),
        (4_096, 10_000),
        (1_048_576, 64),
        (16_777_216, 4),
    ] {
        let mut emulator = Emulator::new(EmulatorConfig {
            device_memory_size: size + 64,
            ..EmulatorConfig::default()
        });
        let input: Vec<u8> = (0..size).map(|i| i.to_le_bytes()[0]).collect();
        emulator.load_device_memory(32, &input)?;
        assert_eq!(emulator.read_device_memory(32, size)?, input);

        let mut times = Vec::with_capacity(7);
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..repetitions {
                black_box(black_box(&emulator).read_device_memory(black_box(32), black_box(size))?);
            }
            times.push(start.elapsed() / repetitions);
        }
        times.sort_unstable();
        let median = times[3];
        let bytes = u32::try_from(size).expect("benchmark sizes fit u32");
        println!(
            "{size},{},{:.2}",
            median.as_nanos(),
            f64::from(bytes) / 1_048_576.0 / median.as_secs_f64()
        );
    }
    Ok(())
}
