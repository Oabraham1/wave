// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for memory operation codegen. Verifies global and shared memory loads,
//!
//! stores, and atomic operations emit correct PTX instructions with proper address
//! computation (64-bit for global, 32-bit for shared) and memory space qualifiers.

use wave_ptx::compile;

const OPCODE_SHIFT: u32 = 26;
const RD_SHIFT: u32 = 21;
const RS1_SHIFT: u32 = 16;
const RS2_SHIFT: u32 = 11;
const MODIFIER_SHIFT: u32 = 7;
const SCOPE_SHIFT: u32 = 5;
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

fn encode_with_scope(
    opcode: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    modifier: u8,
    scope: u8,
    flags: u8,
) -> [u8; 4] {
    let word = ((u32::from(opcode) & 0x3F) << OPCODE_SHIFT)
        | ((u32::from(rd) & 0x1F) << RD_SHIFT)
        | ((u32::from(rs1) & 0x1F) << RS1_SHIFT)
        | ((u32::from(rs2) & 0x1F) << RS2_SHIFT)
        | ((u32::from(modifier) & 0x0F) << MODIFIER_SHIFT)
        | ((u32::from(scope) & 0x03) << SCOPE_SHIFT)
        | (u32::from(flags) & 0x03);
    word.to_le_bytes()
}

fn halt() -> [u8; 4] {
    encode(0x3F, 0, 0, 0, 1, SYNC_OP_FLAG)
}

fn build_wbin(name: &str, reg_count: u32, local_mem: u32, code: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    let name_bytes: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
    let code_offset: u32 = 0x20;
    let code_size = code.len() as u32;
    let symbol_offset = code_offset + code_size;
    let symbol_size = name_bytes.len() as u32;
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
    data.extend_from_slice(&name_bytes);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&symbol_offset.to_le_bytes());
    data.extend_from_slice(&reg_count.to_le_bytes());
    data.extend_from_slice(&local_mem.to_le_bytes());
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
    let wbin = build_wbin("test_kernel", 16, 0, &code);
    compile(&wbin, 75).expect("compilation failed")
}

fn compile_instrs_with_shared(instructions: &[u8], shared_size: u32) -> String {
    let mut code = instructions.to_vec();
    code.extend_from_slice(&halt());
    let wbin = build_wbin("test_kernel", 16, shared_size, &code);
    compile(&wbin, 75).expect("compilation failed")
}

#[test]
fn test_device_load_u32() {
    let ptx = compile_instrs(&encode(0x38, 5, 3, 0, 2, 0));
    assert!(ptx.contains("cvt.u64.u32 %rd1, %r3;"), "PTX: {ptx}");
    assert!(ptx.contains("add.u64 %rd1, %rd0, %rd1;"), "PTX: {ptx}");
    assert!(ptx.contains("ld.global.u32 %r5, [%rd1];"), "PTX: {ptx}");
}

#[test]
fn test_device_store_u32() {
    let ptx = compile_instrs(&encode(0x39, 0, 3, 5, 2, 0));
    assert!(ptx.contains("cvt.u64.u32 %rd1, %r3;"), "PTX: {ptx}");
    assert!(ptx.contains("add.u64 %rd1, %rd0, %rd1;"), "PTX: {ptx}");
    assert!(ptx.contains("st.global.u32 [%rd1], %r5;"), "PTX: {ptx}");
}

#[test]
fn test_device_load_u8() {
    let ptx = compile_instrs(&encode(0x38, 5, 3, 0, 0, 0));
    assert!(ptx.contains("ld.global.u8 %r5, [%rd1];"), "PTX: {ptx}");
}

#[test]
fn test_shared_load_u32() {
    let ptx = compile_instrs_with_shared(&encode(0x30, 5, 3, 0, 2, 0), 4096);
    assert!(ptx.contains("mov.u32 %t0, _shared_mem;"), "PTX: {ptx}");
    assert!(ptx.contains("add.u32 %t0, %t0, %r3;"), "PTX: {ptx}");
    assert!(ptx.contains("ld.shared.u32 %r5, [%t0];"), "PTX: {ptx}");
}

#[test]
fn test_shared_store_u32() {
    let ptx = compile_instrs_with_shared(&encode(0x31, 0, 3, 5, 2, 0), 4096);
    assert!(ptx.contains("st.shared.u32 [%t0], %r5;"), "PTX: {ptx}");
}

#[test]
fn test_device_atomic_add() {
    let mut code = encode_with_scope(0x3D, 5, 3, 4, 0, 2, 0).to_vec();
    code.extend_from_slice(&[0u8; 4]);
    let ptx = compile_instrs(&code);
    assert!(
        ptx.contains("atom.global.add.u32 %r5, [%rd1], %r4;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_device_atomic_cas() {
    let mut code = encode_with_scope(0x3D, 5, 3, 4, 8, 2, 0).to_vec();
    let word1 = (6u32 & 0x1F) << 27;
    code.extend_from_slice(&word1.to_le_bytes());
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("atom.global.cas.b32"), "PTX: {ptx}");
}

#[test]
fn test_fence_cta() {
    let ptx = compile_instrs(&encode_with_scope(0x3F, 0, 0, 0, 3, 1, SYNC_OP_FLAG));
    assert!(ptx.contains("membar.cta;"), "PTX: {ptx}");
}

#[test]
fn test_fence_gl() {
    let ptx = compile_instrs(&encode_with_scope(0x3F, 0, 0, 0, 3, 2, SYNC_OP_FLAG));
    assert!(ptx.contains("membar.gl;"), "PTX: {ptx}");
}
