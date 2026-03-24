// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Test harness for WAVE specification verification. Assembles WAVE programs,
//!
//! runs them on the emulator, and verifies results against expected values.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TestError {
    #[error("assembly failed: {0}")]
    AssemblyFailed(String),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("verification failed: {0}")]
    VerificationFailed(String),
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct EmulatorConfig {
    pub wave_width: u32,
    pub local_memory_size: u32,
    pub device_memory_size: u32,
    pub max_cycles: u64,
    pub max_instructions: u64,
    pub trace: bool,
}

impl Default for EmulatorConfig {
    fn default() -> Self {
        Self {
            wave_width: 4,
            local_memory_size: 16384,
            device_memory_size: 65536,
            max_cycles: 100_000,
            max_instructions: 100_000,
            trace: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub spec_section: String,
    pub spec_claim: String,
    pub passed: bool,
    pub details: String,
    pub cycles: u64,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub device_memory: Vec<u8>,
    pub cycles: u64,
    pub instructions_executed: u64,
}

pub fn assemble(source: &str, source_name: &str) -> Result<Vec<u8>, TestError> {
    wave_asm::assemble(source, source_name)
        .map(|r| r.binary)
        .map_err(|e| TestError::AssemblyFailed(e.to_string()))
}

pub fn run_kernel(
    binary: &[u8],
    grid: [u32; 3],
    workgroup: [u32; 3],
    config: &EmulatorConfig,
    initial_device_memory: Option<&[u8]>,
) -> Result<ExecutionResult, TestError> {
    use wave_emu::{Emulator, EmulatorConfig as EmuConfig};

    let emu_config = EmuConfig {
        grid_dim: grid,
        workgroup_dim: workgroup,
        register_count: 32,
        local_memory_size: config.local_memory_size as usize,
        device_memory_size: config.device_memory_size as usize,
        wave_width: config.wave_width,
        trace_enabled: config.trace,
        f64_enabled: true,
        max_instructions: config.max_instructions,
    };

    let mut emulator = Emulator::new(emu_config);

    emulator
        .load_binary(binary)
        .map_err(|e| TestError::ExecutionFailed(e.to_string()))?;

    if let Some(init_mem) = initial_device_memory {
        emulator
            .load_device_memory(0, init_mem)
            .map_err(|e| TestError::ExecutionFailed(e.to_string()))?;
    }

    let result = emulator
        .run()
        .map_err(|e| TestError::ExecutionFailed(e.to_string()))?;

    let device_memory = emulator
        .read_device_memory(0, config.device_memory_size as usize)
        .map_err(|e| TestError::ExecutionFailed(e.to_string()))?;

    Ok(ExecutionResult {
        device_memory,
        cycles: result.stats.instructions_executed,
        instructions_executed: result.stats.instructions_executed,
    })
}

pub fn run_test(
    asm_source: &str,
    grid: [u32; 3],
    workgroup: [u32; 3],
    config: &EmulatorConfig,
    initial_device_memory: Option<&[u8]>,
) -> Result<ExecutionResult, TestError> {
    let binary = assemble(asm_source, "test.wave")?;
    run_kernel(&binary, grid, workgroup, config, initial_device_memory)
}

pub fn read_u8(memory: &[u8], offset: usize) -> u8 {
    memory.get(offset).copied().unwrap_or(0)
}

pub fn read_u16(memory: &[u8], offset: usize) -> u16 {
    if offset + 2 > memory.len() {
        return 0;
    }
    u16::from_le_bytes([memory[offset], memory[offset + 1]])
}

pub fn read_u32(memory: &[u8], offset: usize) -> u32 {
    if offset + 4 > memory.len() {
        return 0;
    }
    u32::from_le_bytes([
        memory[offset],
        memory[offset + 1],
        memory[offset + 2],
        memory[offset + 3],
    ])
}

pub fn read_u64(memory: &[u8], offset: usize) -> u64 {
    if offset + 8 > memory.len() {
        return 0;
    }
    u64::from_le_bytes([
        memory[offset],
        memory[offset + 1],
        memory[offset + 2],
        memory[offset + 3],
        memory[offset + 4],
        memory[offset + 5],
        memory[offset + 6],
        memory[offset + 7],
    ])
}

pub fn read_f32(memory: &[u8], offset: usize) -> f32 {
    f32::from_bits(read_u32(memory, offset))
}

pub fn read_f64(memory: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64(memory, offset))
}

pub fn check_u32(memory: &[u8], offset: usize, expected: u32) -> bool {
    read_u32(memory, offset) == expected
}

pub fn check_f32(memory: &[u8], offset: usize, expected: f32, tolerance: f32) -> bool {
    let actual = read_f32(memory, offset);
    if expected.is_nan() {
        return actual.is_nan();
    }
    if expected.is_infinite() {
        return actual.is_infinite() && actual.signum() == expected.signum();
    }
    (actual - expected).abs() <= tolerance
}

pub fn check_slice_u32(memory: &[u8], offset: usize, expected: &[u32]) -> bool {
    for (i, &exp) in expected.iter().enumerate() {
        if read_u32(memory, offset + i * 4) != exp {
            return false;
        }
    }
    true
}

pub fn format_memory_u32(memory: &[u8], offset: usize, count: usize) -> String {
    let mut result = String::new();
    for i in 0..count {
        if i > 0 {
            result.push_str(", ");
        }
        result.push_str(&format!("0x{:08x}", read_u32(memory, offset + i * 4)));
    }
    result
}

#[macro_export]
macro_rules! spec_test {
    (
        name: $name:expr,
        section: $section:expr,
        claim: $claim:expr,
        source: $source:expr,
        grid: $grid:expr,
        workgroup: $workgroup:expr,
        check: $check:expr
    ) => {{
        let config = EmulatorConfig::default();
        match run_test($source, $grid, $workgroup, &config, None) {
            Ok(result) => {
                let (passed, details) = $check(&result);
                TestResult {
                    name: $name.to_string(),
                    spec_section: $section.to_string(),
                    spec_claim: $claim.to_string(),
                    passed,
                    details,
                    cycles: result.cycles,
                }
            }
            Err(e) => TestResult {
                name: $name.to_string(),
                spec_section: $section.to_string(),
                spec_claim: $claim.to_string(),
                passed: false,
                details: format!("Execution error: {}", e),
                cycles: 0,
            },
        }
    }};
}

pub const COMPLETION_FLAG: u32 = 0xDEADBEEF;

pub fn check_completion(memory: &[u8], offset: usize) -> bool {
    read_u32(memory, offset) == COMPLETION_FLAG
}
