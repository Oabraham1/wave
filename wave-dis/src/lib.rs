// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE disassembler library. Provides public API for converting WAVE binary files into human-
//! readable assembly text. Delegates all decoding to wave-decode and focuses on text formatting
//! and output generation.

pub mod format;

use format::{format_offset_only, format_with_offset};
use thiserror::Error;
use wave_decode::{DecodeError, Decoder, KernelInfo, WbinError, WbinFile};

#[derive(Debug, Error)]
pub enum DisassembleError {
    #[error("failed to parse WBIN: {0}")]
    WbinParse(#[from] WbinError),
    #[error("failed to decode instruction: {0}")]
    Decode(#[from] DecodeError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug)]
pub struct DisassemblyOptions {
    pub show_offsets: bool,
    pub show_raw: bool,
    pub emit_directives: bool,
}

impl Default for DisassemblyOptions {
    fn default() -> Self {
        Self {
            show_offsets: false,
            show_raw: false,
            emit_directives: true,
        }
    }
}

#[derive(Debug)]
pub struct DisassembledLine {
    pub offset: Option<u32>,
    pub text: String,
}

/// Disassemble a WBIN file into assembly text lines.
///
/// # Errors
///
/// Returns `DisassembleError` if the WBIN parsing or instruction decoding fails.
pub fn disassemble_wbin(
    data: &[u8],
    options: &DisassemblyOptions,
) -> Result<Vec<String>, DisassembleError> {
    let wbin = WbinFile::parse(data)?;
    let code = wbin.code();
    let code_base = wbin.header.code_offset;

    let mut lines = Vec::new();

    if wbin.kernels.is_empty() {
        append_code(&mut lines, code, code_base, options)?;
    } else {
        for (i, kernel) in wbin.kernels.iter().enumerate() {
            if options.emit_directives {
                lines.extend(format_kernel_header(kernel));
            }

            if let Some(kernel_code) = wbin.kernel_code(i) {
                let kernel_base = code_base + kernel.code_offset;
                append_code(&mut lines, kernel_code, kernel_base, options)?;
            }

            if options.emit_directives {
                lines.push(".end".to_string());
                if i < wbin.kernels.len() - 1 {
                    lines.push(String::new());
                }
            }
        }
    }

    Ok(lines)
}

/// Disassemble raw code bytes into assembly text lines.
///
/// # Errors
///
/// Returns `DisassembleError` if instruction decoding fails.
pub fn disassemble_code(
    code: &[u8],
    options: &DisassemblyOptions,
) -> Result<Vec<String>, DisassembleError> {
    let mut lines = Vec::new();
    append_code(&mut lines, code, 0, options)?;
    Ok(lines)
}

fn append_code(
    lines: &mut Vec<String>,
    code: &[u8],
    code_base: u32,
    options: &DisassemblyOptions,
) -> Result<(), DecodeError> {
    let mut decoder = Decoder::new(code);
    while decoder.has_more() {
        let inst = decoder.decode_next()?;
        lines.push(format_line(&inst, code, code_base, options));
    }
    Ok(())
}

fn format_kernel_header(kernel: &KernelInfo) -> Vec<String> {
    let mut lines = Vec::new();

    if kernel.name.is_empty() {
        lines.push(".kernel unnamed".to_string());
    } else {
        lines.push(format!(".kernel {}", kernel.name));
    }

    if kernel.register_count > 0 {
        lines.push(format!(".registers {}", kernel.register_count));
    }

    let [x, y, z] = kernel.workgroup_size;
    if x != 1 || y != 1 || z != 1 {
        lines.push(format!(".workgroup_size {x}, {y}, {z}"));
    }

    if kernel.local_memory_size > 0 {
        lines.push(format!(".local_memory {}", kernel.local_memory_size));
    }

    lines
}

fn format_line(
    inst: &wave_decode::DecodedInstruction,
    code: &[u8],
    code_base: u32,
    options: &DisassemblyOptions,
) -> String {
    if options.show_offsets && options.show_raw {
        format_with_offset(inst, code, code_base)
    } else if options.show_offsets {
        format_offset_only(inst, code_base)
    } else {
        format!("    {}", format::format_instruction(inst))
    }
}

pub use format::{format_instruction, FormatOptions};
