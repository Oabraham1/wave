// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate WBIN files or individual WAVE kernels into Metal.

pub mod codegen;
pub mod control_flow;
pub mod intrinsics;
pub mod kernel;
pub mod memory;
pub mod registers;

use thiserror::Error;
use wave_decode::{DecodeError, KernelInfo, WbinError, WbinFile};

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("WBIN parse error: {0}")]
    Wbin(#[from] WbinError),

    #[error("decode error: {0}")]
    Decode(#[from] DecodeError),

    #[error("no kernels found in WBIN file")]
    NoKernels,

    #[error("invalid kernel index: {0}")]
    InvalidKernel(usize),

    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Compile a complete WBIN file into Metal Shading Language source.
///
/// # Errors
///
/// Returns `CompileError` if the WBIN file cannot be parsed, contains no kernels
/// or contains unsupported instructions.
pub fn compile(wbin_data: &[u8]) -> Result<String, CompileError> {
    compile_with_initial_registers(wbin_data, &[])
}

/// Compile WBIN with initial values for consecutive argument registers.
///
/// Registers beyond the provided arguments retain their zero initialization.
/// Arguments for registers unused by a kernel are ignored.
///
/// # Errors
///
/// Returns an error for invalid WBIN, unsupported instructions, or more than
/// 256 initial register values.
///
/// # Examples
///
/// ```
/// assert!(wave_metal::compile_with_initial_registers(&[], &[0, 64]).is_err());
/// ```
pub fn compile_with_initial_registers(
    wbin_data: &[u8],
    initial_registers: &[u32],
) -> Result<String, CompileError> {
    let wbin = WbinFile::parse(wbin_data)?;
    if wbin.kernels.is_empty() {
        return Err(CompileError::NoKernels);
    }

    let mut output = kernel::emit_file_header();

    for (i, kernel_info) in wbin.kernels.iter().enumerate() {
        let code = wbin.kernel_code(i).ok_or(CompileError::InvalidKernel(i))?;
        let instructions = wave_decode::decode_all(code)?;

        let mut gen = codegen::CodeGenerator::new();
        let kernel_msl =
            gen.generate_with_initial_registers(&instructions, kernel_info, initial_registers)?;
        output.push_str(&kernel_msl);
        output.push('\n');
    }

    Ok(output)
}

/// Compile a single kernel's code section into an MSL kernel function.
///
/// # Errors
///
/// Returns `CompileError` if the code cannot be decoded or contains unsupported instructions.
pub fn compile_kernel(code: &[u8], kernel: &KernelInfo) -> Result<String, CompileError> {
    let instructions = wave_decode::decode_all(code)?;
    let mut gen = codegen::CodeGenerator::new();
    let mut output = kernel::emit_file_header();
    output.push_str(&gen.generate(&instructions, kernel)?);
    Ok(output)
}
