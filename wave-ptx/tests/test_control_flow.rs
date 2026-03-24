// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for control flow lowering. Verifies that WAVE structured control flow
//!
//! (if/else/endif, loop/break/continue/endloop) is correctly lowered to PTX predicated
//! branches with properly generated and placed labels.

use wave_ptx::compile;

const OPCODE_SHIFT: u32 = 26;
const RD_SHIFT: u32 = 21;
const RS1_SHIFT: u32 = 16;
const RS2_SHIFT: u32 = 11;
const MODIFIER_SHIFT: u32 = 7;
const SYNC_OP_FLAG: u8 = 0x01;

fn encode(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, flags: u8) -> [u8; 4] {
    let word = ((u32::from(opcode) & 0x3F) << OPCODE_SHIFT)
        | ((u32::from(rd) & 0x1F) << RD_SHIFT)
        | ((u32::from(rs1) & 0x1F) << RS1_SHIFT)
        | ((u32::from(rs2) & 0x1F) << RS2_SHIFT)
        | ((u32::from(modifier) & 0x0F) << MODIFIER_SHIFT)
        | (u32::from(flags) & 0x03);
    word.to_le_bytes()
}

fn halt() -> [u8; 4] {
    encode(0x3F, 0, 0, 0, 1, SYNC_OP_FLAG)
}

fn build_wbin(code: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    let name = b"test_kernel\0";
    let code_offset: u32 = 0x20;
    let code_size = code.len() as u32;
    let symbol_offset = code_offset + code_size;
    let symbol_size = name.len() as u32;
    let metadata_offset = symbol_offset + symbol_size;
    let metadata_size: u32 = 4 + 32;

    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&code_offset.to_le_bytes());
    data.extend_from_slice(&code_size.to_le_bytes());
    data.extend_from_slice(&symbol_offset.to_le_bytes());
    data.extend_from_slice(&symbol_size.to_le_bytes());
    data.extend_from_slice(&metadata_offset.to_le_bytes());
    data.extend_from_slice(&metadata_size.to_le_bytes());
    data.extend_from_slice(code);
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&symbol_offset.to_le_bytes());
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&64u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&code_size.to_le_bytes());
    data
}

fn compile_instrs(instructions: &[u8]) -> String {
    let mut code = instructions.to_vec();
    code.extend_from_slice(&halt());
    let wbin = build_wbin(&code);
    compile(&wbin, 75).expect("compilation failed")
}

#[test]
fn test_if_endif() {
    let mut code = Vec::new();
    code.extend_from_slice(&encode(0x3F, 0, 1, 0, 0, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x00, 5, 3, 4, 0, 0));
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 2, 0));
    code.extend_from_slice(&[0u8; 4]);
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("@!%p1 bra $L_else_0;"), "PTX: {ptx}");
    assert!(ptx.contains("add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
    assert!(ptx.contains("$L_else_0:"), "PTX: {ptx}");
}

#[test]
fn test_if_else_endif() {
    let mut code = Vec::new();
    code.extend_from_slice(&encode(0x3F, 0, 1, 0, 0, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x00, 5, 3, 4, 0, 0));
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 1, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x01, 5, 3, 4, 0, 0));
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 2, 0));
    code.extend_from_slice(&[0u8; 4]);
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("@!%p1 bra $L_else_0;"), "PTX: {ptx}");
    assert!(ptx.contains("bra.uni $L_endif_0;"), "PTX: {ptx}");
    assert!(ptx.contains("$L_else_0:"), "PTX: {ptx}");
    assert!(ptx.contains("$L_endif_0:"), "PTX: {ptx}");
    assert!(ptx.contains("add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
    assert!(ptx.contains("sub.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_loop_break() {
    let mut code = Vec::new();
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 3, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x3F, 0, 1, 0, 4, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 6, 0));
    code.extend_from_slice(&[0u8; 4]);
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("$L_loop_0:"), "PTX: {ptx}");
    assert!(ptx.contains("@%p1 bra $L_endloop_0;"), "PTX: {ptx}");
    assert!(ptx.contains("bra.uni $L_loop_0;"), "PTX: {ptx}");
    assert!(ptx.contains("$L_endloop_0:"), "PTX: {ptx}");
}

#[test]
fn test_loop_continue() {
    let mut code = Vec::new();
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 3, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x3F, 0, 2, 0, 5, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 6, 0));
    code.extend_from_slice(&[0u8; 4]);
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("@%p2 bra $L_loop_0;"), "PTX: {ptx}");
}
