// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate WBIN files or individual WAVE kernels into PTX.

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

    /// Structured control flow has unmatched or misplaced instructions.
    #[error("invalid control flow: {0}")]
    InvalidControlFlow(&'static str),

    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Compile a complete WBIN file into PTX assembly text.
///
/// # Errors
///
/// Returns `CompileError` if the WBIN file cannot be parsed, contains no kernels
/// or contains unsupported instructions.
pub fn compile(wbin_data: &[u8], sm_version: u32) -> Result<String, CompileError> {
    let wbin = WbinFile::parse(wbin_data)?;
    if wbin.kernels.is_empty() {
        return Err(CompileError::NoKernels);
    }

    let mut output = kernel::emit_header(sm_version);

    for (i, kernel_info) in wbin.kernels.iter().enumerate() {
        let code = wbin.kernel_code(i).ok_or(CompileError::InvalidKernel(i))?;
        let instructions = wave_decode::decode_all(code)?;

        let mut gen = codegen::CodeGenerator::new();
        let kernel_ptx = gen.generate(&instructions, kernel_info)?;
        output.push_str(&kernel_ptx);
        output.push('\n');
    }

    Ok(output)
}

/// Compile a single kernel's code section into a PTX entry function.
///
/// # Errors
///
/// Returns `CompileError` if the code cannot be decoded or contains unsupported instructions.
pub fn compile_kernel(
    code: &[u8],
    kernel: &KernelInfo,
    sm_version: u32,
) -> Result<String, CompileError> {
    let instructions = wave_decode::decode_all(code)?;
    let mut gen = codegen::CodeGenerator::new();
    let mut output = kernel::emit_header(sm_version);
    output.push_str(&gen.generate(&instructions, kernel)?);
    Ok(output)
}
