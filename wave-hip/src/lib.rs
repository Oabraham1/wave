// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Public API for the WAVE HIP backend. Provides compile() to translate a complete
//!
//! WBIN binary file into HIP C++ source text, and compile_kernel() for translating
//! a single kernel's code section. Targets AMD GPUs via ROCm/HIP with support for
//! both RDNA (wavefront 32) and CDNA (wavefront 64) architectures.

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

/// Compile a complete WBIN file into HIP C++ source.
///
/// # Errors
///
/// Returns `CompileError` if the WBIN file cannot be parsed, contains no kernels,
/// or contains unsupported instructions.
pub fn compile(wbin_data: &[u8]) -> Result<String, CompileError> {
    let wbin = WbinFile::parse(wbin_data)?;
    if wbin.kernels.is_empty() {
        return Err(CompileError::NoKernels);
    }

    let mut output = kernel::emit_file_header();

    for (i, kernel_info) in wbin.kernels.iter().enumerate() {
        let code = wbin
            .kernel_code(i)
            .ok_or(CompileError::InvalidKernel(i))?;
        let instructions = wave_decode::decode_all(code)?;

        let mut gen = codegen::CodeGenerator::new();
        let kernel_hip = gen.generate(&instructions, kernel_info)?;
        output.push_str(&kernel_hip);
        output.push('\n');
    }

    Ok(output)
}

/// Compile a single kernel's code section into a HIP kernel function.
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
