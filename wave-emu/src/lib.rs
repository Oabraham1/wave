// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Load and execute WAVE binaries with configurable dispatch dimensions and execution
//! statistics.

pub mod barrier;
pub mod control_flow;
pub mod core;
pub mod decoder;
pub mod executor;
pub mod memory;
pub mod scheduler;
pub mod shuffle;
pub mod stats;
pub mod thread;
pub mod wave;

use crate::core::Core;
use crate::memory::DeviceMemory;
use crate::stats::ExecutionStats;
use std::io::Read;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmulatorError {
    #[error("invalid WBIN file: {message}")]
    InvalidBinary { message: String },

    #[error("memory access out of bounds: address 0x{address:016x}")]
    MemoryOutOfBounds { address: u64 },

    #[error("invalid instruction at PC 0x{pc:08x}: {message}")]
    InvalidInstruction { pc: u32, message: String },

    #[error("control flow error: {message}")]
    ControlFlowError { message: String },

    #[error("deadlock detected: {message}")]
    Deadlock { message: String },

    #[error("division by zero")]
    DivisionByZero,

    #[error("stack overflow: {kind}")]
    StackOverflow { kind: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error(
        "instruction limit exceeded: {executed} instructions, limit {limit}, at PC 0x{pc:08x}"
    )]
    InstructionLimitExceeded { limit: u64, executed: u64, pc: u32 },
}

#[derive(Debug, Clone)]
pub struct EmulatorConfig {
    pub grid_dim: [u32; 3],
    pub workgroup_dim: [u32; 3],
    /// Use the kernel's declared workgroup size instead of `workgroup_dim`.
    pub use_kernel_workgroup_size: bool,
    pub register_count: u32,
    pub local_memory_size: usize,
    pub device_memory_size: usize,
    pub wave_width: u32,
    pub trace_enabled: bool,
    pub f64_enabled: bool,
    /// Maximum instructions to execute, 0 = unlimited. Default: 10,000,000
    pub max_instructions: u64,
    /// Initial register values set for all threads before execution.
    /// Each pair contains a register index and its initial value.
    pub initial_registers: Vec<(u8, u32)>,
}

impl Default for EmulatorConfig {
    fn default() -> Self {
        Self {
            grid_dim: [1, 1, 1],
            workgroup_dim: [32, 1, 1],
            use_kernel_workgroup_size: true,
            register_count: 32,
            local_memory_size: 16384,
            device_memory_size: 1024 * 1024,
            wave_width: 32,
            trace_enabled: false,
            f64_enabled: false,
            max_instructions: 10_000_000,
            initial_registers: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct EmulatorResult {
    pub stats: ExecutionStats,
}

#[derive(Debug)]
pub struct Emulator {
    config: EmulatorConfig,
    device_memory: DeviceMemory,
    code: Vec<u8>,
    kernel_metadata: Vec<KernelMetadata>,
}

#[derive(Debug, Clone)]
pub struct KernelMetadata {
    pub name: String,
    pub register_count: u32,
    pub local_memory_size: u32,
    pub workgroup_size: [u32; 3],
    pub code_offset: u32,
    pub code_size: u32,
}

impl Emulator {
    #[must_use]
    pub fn new(config: EmulatorConfig) -> Self {
        let device_memory = DeviceMemory::new(config.device_memory_size);
        Self {
            config,
            device_memory,
            code: Vec::new(),
            kernel_metadata: Vec::new(),
        }
    }

    /// Load a validated WBIN binary and replace the current kernels.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed header, section, or kernel range.
    pub fn load_binary(&mut self, binary: &[u8]) -> Result<(), EmulatorError> {
        let parsed =
            wave_decode::WbinFile::parse(binary).map_err(|error| EmulatorError::InvalidBinary {
                message: error.to_string(),
            })?;
        for (index, _) in parsed.kernels.iter().enumerate() {
            if parsed.kernel_code(index).is_none() {
                return Err(EmulatorError::InvalidBinary {
                    message: format!("kernel {index} extends beyond the code section"),
                });
            }
        }
        let kernels = parsed
            .kernels
            .iter()
            .enumerate()
            .map(|(index, kernel)| KernelMetadata {
                name: if kernel.name.is_empty() {
                    format!("kernel_{index}")
                } else {
                    kernel.name.clone()
                },
                register_count: kernel.register_count,
                local_memory_size: kernel.local_memory_size,
                workgroup_size: kernel.workgroup_size,
                code_offset: kernel.code_offset,
                code_size: kernel.code_size,
            })
            .collect();
        self.code = parsed.code().to_vec();
        self.kernel_metadata = kernels;
        Ok(())
    }

    /// Copy bytes into device memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination range is outside device memory.
    pub fn load_device_memory(&mut self, offset: u64, data: &[u8]) -> Result<(), EmulatorError> {
        self.device_memory.write_slice(offset, data)
    }

    /// Copy a range of device memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested range is outside device memory.
    pub fn read_device_memory(&self, offset: u64, len: usize) -> Result<Vec<u8>, EmulatorError> {
        Ok(self.device_memory.read_slice(offset, len)?.to_vec())
    }

    /// Run until execution halts.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid instructions, memory accesses, control flow, deadlock, or an exceeded instruction limit.
    pub fn run(&mut self) -> Result<EmulatorResult, EmulatorError> {
        self.run_kernel(0)
    }

    /// Execute a selected kernel.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid instructions, memory accesses, control flow, deadlock, or an exceeded instruction limit.
    pub fn run_kernel(&mut self, kernel_index: usize) -> Result<EmulatorResult, EmulatorError> {
        let mut effective_config = self.config.clone();

        if kernel_index < self.kernel_metadata.len() {
            let meta = &self.kernel_metadata[kernel_index];
            if meta.register_count > 0 {
                effective_config.register_count = meta.register_count;
            }
            if meta.local_memory_size > 0 {
                effective_config.local_memory_size = meta.local_memory_size as usize;
            }
            if effective_config.use_kernel_workgroup_size && meta.workgroup_size[0] > 0 {
                effective_config.workgroup_dim = meta.workgroup_size;
            }
        }

        let code_start = if kernel_index < self.kernel_metadata.len() {
            self.kernel_metadata[kernel_index].code_offset as usize
        } else {
            0
        };

        let mut total_stats = ExecutionStats::default();

        for wg_z in 0..effective_config.grid_dim[2] {
            for wg_y in 0..effective_config.grid_dim[1] {
                for wg_x in 0..effective_config.grid_dim[0] {
                    let mut core = Core::new(
                        &effective_config,
                        &self.code[code_start..],
                        &mut self.device_memory,
                        [wg_x, wg_y, wg_z],
                    );

                    let stats = core.run()?;
                    total_stats.merge(&stats);
                    total_stats.workgroups_executed += 1;
                }
            }
        }

        Ok(EmulatorResult { stats: total_stats })
    }

    #[must_use]
    pub fn device_memory(&self) -> &DeviceMemory {
        &self.device_memory
    }

    pub fn device_memory_mut(&mut self) -> &mut DeviceMemory {
        &mut self.device_memory
    }

    #[must_use]
    pub fn kernels(&self) -> &[KernelMetadata] {
        &self.kernel_metadata
    }
}

/// Read a WBIN file from disk.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub fn load_binary_file(path: &std::path::Path) -> Result<Vec<u8>, EmulatorError> {
    let mut file = std::fs::File::open(path).map_err(|e| EmulatorError::IoError {
        message: e.to_string(),
    })?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| EmulatorError::IoError {
            message: e.to_string(),
        })?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_replaces_kernel_metadata() {
        let first = wave_asm::assemble(".kernel first\nhalt\n.end", "test.wave")
            .unwrap()
            .binary;
        let second = wave_asm::assemble(".kernel second\nhalt\n.end", "test.wave")
            .unwrap()
            .binary;
        let mut emulator = Emulator::new(EmulatorConfig::default());
        emulator.load_binary(&first).unwrap();
        emulator.load_binary(&second).unwrap();
        assert_eq!(emulator.kernels().len(), 1);
        assert_eq!(emulator.kernels()[0].name, "second");
    }

    #[test]
    fn truncated_metadata_returns_an_error() {
        let mut binary = wave_asm::assemble(".kernel test\nhalt\n.end", "test.wave")
            .unwrap()
            .binary;
        binary.truncate(binary.len() - 1);
        let mut emulator = Emulator::new(EmulatorConfig::default());
        assert!(emulator.load_binary(&binary).is_err());
        assert!(emulator.read_device_memory(u64::MAX, 2).is_err());
    }
}
