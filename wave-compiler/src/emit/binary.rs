// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE binary (.wbin) file generation.
//!
//! Assembles encoded WAVE instructions into the WBIN container format
//! with header, code section, symbol table, and kernel metadata.

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
    let mut code_bytes = Vec::new();
    for inst in instructions {
        let encoded = emit_instruction(inst, reg_map);
        code_bytes.extend_from_slice(&encoded.to_bytes());
    }

    let code_offset = WBIN_HEADER_SIZE;
    let code_size = code_bytes.len() as u32;

    let kernel_name_bytes = kernel_name.as_bytes();
    let symbol_offset = code_offset + code_size;
    let symbol_size = (kernel_name_bytes.len() + 1) as u32;

    let metadata_offset = symbol_offset + symbol_size;
    let metadata_size = 4 + KERNEL_METADATA_SIZE;

    let mut output = Vec::new();

    output.extend_from_slice(WBIN_MAGIC);
    output.extend_from_slice(&WBIN_VERSION.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output.extend_from_slice(&code_offset.to_le_bytes());
    output.extend_from_slice(&code_size.to_le_bytes());
    output.extend_from_slice(&symbol_offset.to_le_bytes());
    output.extend_from_slice(&symbol_size.to_le_bytes());
    output.extend_from_slice(&metadata_offset.to_le_bytes());
    output.extend_from_slice(&metadata_size.to_le_bytes());

    output.extend_from_slice(&code_bytes);

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
        for src in inst.src_vregs() {
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
