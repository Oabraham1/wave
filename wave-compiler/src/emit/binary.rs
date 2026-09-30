// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WBIN file generation. Assembles encoded WAVE instructions into the WBIN
//! container format with header, code section, symbol table, and kernel metadata.

use crate::diagnostics::CompileError;
use crate::emit::wave_emit::{emit_instruction, RegMap};
use crate::lir::instruction::LirInst;

const WBIN_MAGIC: &[u8; 4] = b"WAVE";
const WBIN_VERSION: u16 = 0x0001;
const WBIN_HEADER_SIZE: u32 = 32;
const KERNEL_METADATA_SIZE: u32 = 32;

/// Generate a complete .wbin binary from LIR instructions.
///
/// # Errors
///
/// Returns `CompileError` if binary generation fails.
pub fn generate_wbin(
    kernel_name: &str,
    instructions: &[LirInst],
    reg_map: &RegMap,
    register_count: u32,
) -> Result<Vec<u8>, CompileError> {
    let mut output = vec![0; WBIN_HEADER_SIZE as usize];
    for inst in instructions {
        emit_instruction(inst, reg_map)?.append_to(&mut output);
    }

    let code_offset = WBIN_HEADER_SIZE;
    let code_size = binary_size(output.len() - WBIN_HEADER_SIZE as usize)?;

    let kernel_name_bytes = kernel_name.as_bytes();
    let symbol_offset = code_offset
        .checked_add(code_size)
        .ok_or_else(binary_too_large)?;
    let symbol_size = binary_size(kernel_name_bytes.len())?
        .checked_add(1)
        .ok_or_else(binary_too_large)?;

    let metadata_offset = symbol_offset
        .checked_add(symbol_size)
        .ok_or_else(binary_too_large)?;
    let metadata_size = 4 + KERNEL_METADATA_SIZE;

    output[0..4].copy_from_slice(WBIN_MAGIC);
    output[4..6].copy_from_slice(&WBIN_VERSION.to_le_bytes());
    for (bytes, value) in output[8..32].as_chunks_mut::<4>().0.iter_mut().zip([
        code_offset,
        code_size,
        symbol_offset,
        symbol_size,
        metadata_offset,
        metadata_size,
    ]) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }

    output.extend_from_slice(kernel_name_bytes);
    output.push(0);

    output.extend_from_slice(&1u32.to_le_bytes());

    output.extend_from_slice(&symbol_offset.to_le_bytes());
    output.extend_from_slice(&register_count.to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes());
    output.extend_from_slice(&256u32.to_le_bytes());
    output.extend_from_slice(&1u32.to_le_bytes());
    output.extend_from_slice(&1u32.to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes());
    output.extend_from_slice(&code_size.to_le_bytes());

    Ok(output)
}

fn binary_too_large() -> CompileError {
    CompileError::CodegenError {
        message: "binary exceeds the WBIN 32-bit size limit".into(),
    }
}

fn binary_size(size: usize) -> Result<u32, CompileError> {
    u32::try_from(size).map_err(|_| binary_too_large())
}

/// Count the maximum physical register used in the LIR instructions.
#[must_use]
pub fn count_registers(instructions: &[LirInst], reg_map: &RegMap) -> u32 {
    let mut max_reg: u32 = 0;
    for inst in instructions {
        if let Some(dest) = inst.dest_vreg() {
            let phys = reg_map.get(&dest).map_or(dest.0, |p| u32::from(p.0));
            if phys + 1 > max_reg {
                max_reg = phys + 1;
            }
        }
        for src in inst.src_vreg_iter() {
            let phys = reg_map.get(&src).map_or(src.0, |p| u32::from(p.0));
            if phys + 1 > max_reg {
                max_reg = phys + 1;
            }
        }
    }
    max_reg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lir::operand::VReg;
    use std::collections::HashMap;

    #[test]
    fn test_generate_minimal_wbin() {
        let instructions = vec![LirInst::Halt];
        let reg_map = HashMap::new();
        let wbin = generate_wbin("test_kernel", &instructions, &reg_map, 4).unwrap();

        assert_eq!(&wbin[0..4], b"WAVE");
        assert_eq!(u16::from_le_bytes([wbin[4], wbin[5]]), WBIN_VERSION);

        let wbin_file = wave_decode::WbinFile::parse(&wbin).unwrap();
        assert_eq!(wbin_file.kernels.len(), 1);
        assert_eq!(wbin_file.kernels[0].name, "test_kernel");
    }

    #[test]
    fn test_generate_wbin_with_instructions() {
        let instructions = vec![
            LirInst::MovImm {
                dest: VReg(0),
                value: 42,
            },
            LirInst::Halt,
        ];
        let reg_map = HashMap::new();
        let wbin = generate_wbin("my_kernel", &instructions, &reg_map, 1).unwrap();

        let wbin_file = wave_decode::WbinFile::parse(&wbin).unwrap();
        let code = wbin_file.code();
        assert!(code.len() > 4);

        let decoded = wave_decode::decode_all(code).unwrap();
        assert_eq!(decoded.len(), 2);
    }

    #[test]
    fn test_count_registers() {
        let instructions = vec![
            LirInst::Iadd {
                dest: VReg(5),
                src1: VReg(3),
                src2: VReg(4),
            },
            LirInst::Halt,
        ];
        let reg_map = HashMap::new();
        assert_eq!(count_registers(&instructions, &reg_map), 6);
    }
}
