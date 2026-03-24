// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Public API for the WAVE emulator. Provides Emulator struct for running WAVE
//!
//! binaries, configuration options for grid/workgroup dimensions, and execution
//! results with statistics. Entry point for programmatic use of the emulator.

#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::collapsible_if,
    clippy::collapsible_str_replace,
    clippy::comparison_to_empty,
    clippy::float_cmp,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::needless_return,
    clippy::redundant_closure_for_method_calls,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::struct_excessive_bools,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

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

    #[error("instruction limit exceeded: {executed} instructions (limit: {limit}) at PC 0x{pc:08x}")]
    InstructionLimitExceeded { limit: u64, executed: u64, pc: u32 },
}

#[derive(Debug, Clone)]
pub struct EmulatorConfig {
    pub grid_dim: [u32; 3],
    pub workgroup_dim: [u32; 3],
    pub register_count: u32,
    pub local_memory_size: usize,
    pub device_memory_size: usize,
    pub wave_width: u32,
    pub trace_enabled: bool,
    pub f64_enabled: bool,
    /// Maximum instructions to execute (0 = unlimited). Default: 10,000,000
    pub max_instructions: u64,
}

impl Default for EmulatorConfig {
    fn default() -> Self {
        Self {
            grid_dim: [1, 1, 1],
            workgroup_dim: [32, 1, 1],
            register_count: 32,
            local_memory_size: 16384,
            device_memory_size: 1024 * 1024,
            wave_width: 32,
            trace_enabled: false,
            f64_enabled: false,
            max_instructions: 10_000_000,
        }
    }
}

#[derive(Debug)]
pub struct EmulatorResult {
    pub stats: ExecutionStats,
}

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
    pub fn new(config: EmulatorConfig) -> Self {
        let device_memory = DeviceMemory::new(config.device_memory_size);
        Self {
            config,
            device_memory,
            code: Vec::new(),
            kernel_metadata: Vec::new(),
        }
    }

    pub fn load_binary(&mut self, binary: &[u8]) -> Result<(), EmulatorError> {
        if binary.len() < 0x20 {
            return Err(EmulatorError::InvalidBinary {
                message: "file too small for WBIN header".into(),
            });
        }

        if &binary[0..4] != b"WAVE" {
            return Err(EmulatorError::InvalidBinary {
                message: "invalid magic number".into(),
            });
        }

        let code_offset = u32::from_le_bytes([binary[0x08], binary[0x09], binary[0x0A], binary[0x0B]]) as usize;
        let code_size = u32::from_le_bytes([binary[0x0C], binary[0x0D], binary[0x0E], binary[0x0F]]) as usize;
        let symbol_offset = u32::from_le_bytes([binary[0x10], binary[0x11], binary[0x12], binary[0x13]]) as usize;
        let metadata_offset = u32::from_le_bytes([binary[0x18], binary[0x19], binary[0x1A], binary[0x1B]]) as usize;

        if code_offset + code_size > binary.len() {
            return Err(EmulatorError::InvalidBinary {
                message: "code section extends beyond file".into(),
            });
        }

        self.code = binary[code_offset..code_offset + code_size].to_vec();

        if metadata_offset < binary.len() {
            let kernel_count = u32::from_le_bytes([
                binary[metadata_offset],
                binary[metadata_offset + 1],
                binary[metadata_offset + 2],
                binary[metadata_offset + 3],
            ]) as usize;

            for i in 0..kernel_count {
                let base = metadata_offset + 4 + i * 32;
                if base + 32 > binary.len() {
                    break;
                }

                let name_offset = u32::from_le_bytes([
                    binary[base],
                    binary[base + 1],
                    binary[base + 2],
                    binary[base + 3],
                ]) as usize;

                let name = if symbol_offset > 0 && name_offset >= symbol_offset && name_offset < binary.len() {
                    let mut end = name_offset;
                    while end < binary.len() && binary[end] != 0 {
                        end += 1;
                    }
                    String::from_utf8_lossy(&binary[name_offset..end]).to_string()
                } else {
                    format!("kernel_{i}")
                };

                let register_count = u32::from_le_bytes([
                    binary[base + 4],
                    binary[base + 5],
                    binary[base + 6],
                    binary[base + 7],
                ]);

                let local_memory_size = u32::from_le_bytes([
                    binary[base + 8],
                    binary[base + 9],
                    binary[base + 10],
                    binary[base + 11],
                ]);

                let workgroup_size = [
                    u32::from_le_bytes([
                        binary[base + 12],
                        binary[base + 13],
                        binary[base + 14],
                        binary[base + 15],
                    ]),
                    u32::from_le_bytes([
                        binary[base + 16],
                        binary[base + 17],
                        binary[base + 18],
                        binary[base + 19],
                    ]),
                    u32::from_le_bytes([
                        binary[base + 20],
                        binary[base + 21],
                        binary[base + 22],
                        binary[base + 23],
                    ]),
                ];

                let kernel_code_offset = u32::from_le_bytes([
                    binary[base + 24],
                    binary[base + 25],
                    binary[base + 26],
                    binary[base + 27],
                ]);

                let kernel_code_size = u32::from_le_bytes([
                    binary[base + 28],
                    binary[base + 29],
                    binary[base + 30],
                    binary[base + 31],
                ]);

                self.kernel_metadata.push(KernelMetadata {
                    name,
                    register_count,
                    local_memory_size,
                    workgroup_size,
                    code_offset: kernel_code_offset,
                    code_size: kernel_code_size,
                });
            }
        }

        Ok(())
    }

    pub fn load_device_memory(&mut self, offset: u64, data: &[u8]) -> Result<(), EmulatorError> {
        self.device_memory.write_slice(offset, data)
    }

    pub fn read_device_memory(&self, offset: u64, len: usize) -> Result<Vec<u8>, EmulatorError> {
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(self.device_memory.read_u8(offset + i as u64)?);
        }
        Ok(result)
    }

    pub fn run(&mut self) -> Result<EmulatorResult, EmulatorError> {
        self.run_kernel(0)
    }

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
            if meta.workgroup_size[0] > 0 {
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

    pub fn device_memory(&self) -> &DeviceMemory {
        &self.device_memory
    }

    pub fn device_memory_mut(&mut self) -> &mut DeviceMemory {
        &mut self.device_memory
    }

    pub fn kernels(&self) -> &[KernelMetadata] {
        &self.kernel_metadata
    }
}

pub fn load_binary_file(path: &std::path::Path) -> Result<Vec<u8>, EmulatorError> {
    let mut file = std::fs::File::open(path).map_err(|e| EmulatorError::IoError {
        message: e.to_string(),
    })?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| EmulatorError::IoError {
        message: e.to_string(),
    })?;
    Ok(buffer)
}
