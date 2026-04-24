// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for memory operation codegen. Verifies global and shared memory loads,
//!
//! stores, and atomic operations emit correct PTX instructions with proper address
//! computation (64-bit for global, 32-bit for shared) and memory space qualifiers.

use wave_ptx::compile;

const OPCODE_SHIFT: u32 = 24;
const RD_SHIFT: u32 = 16;
const RS1_SHIFT: u32 = 8;
const MODIFIER_SHIFT: u32 = 4;
const EXTENDED_RS2_SHIFT: u32 = 24;
const EXTENDED_RS3_SHIFT: u32 = 16;

fn encode_word0(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred: u8) -> u32 {
    (u32::from(opcode) << OPCODE_SHIFT)
        | (u32::from(rd) << RD_SHIFT)
        | (u32::from(rs1) << RS1_SHIFT)
        | (u32::from(modifier) << MODIFIER_SHIFT)
        | u32::from(pred)
}
fn single(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred: u8) -> Vec<u8> {
    encode_word0(opcode, rd, rs1, modifier, pred)
        .to_le_bytes()
        .to_vec()
}
fn extended(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, pred: u8) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, pred)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(&(u32::from(rs2) << EXTENDED_RS2_SHIFT).to_le_bytes());
    v
}
fn extended_scope(
    opcode: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    modifier: u8,
    scope: u8,
    pred: u8,
) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, pred)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(&((u32::from(rs2) << EXTENDED_RS2_SHIFT) | u32::from(scope)).to_le_bytes());
    v
}
fn extended3_scope(
    opcode: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    rs3: u8,
    modifier: u8,
    scope: u8,
    pred: u8,
) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, pred)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(
        &((u32::from(rs2) << EXTENDED_RS2_SHIFT)
            | (u32::from(rs3) << EXTENDED_RS3_SHIFT)
            | u32::from(scope))
        .to_le_bytes(),
    );
    v
}

fn halt_instruction() -> Vec<u8> {
    single(0x3F, 0, 0, 9, 0)
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
    code.extend_from_slice(&halt_instruction());
    let wbin = build_wbin("test_kernel", 16, 0, &code);
    compile(&wbin, 75).expect("compilation failed")
}

fn compile_instrs_with_shared(instructions: &[u8], shared_size: u32) -> String {
    let mut code = instructions.to_vec();
    code.extend_from_slice(&halt_instruction());
    let wbin = build_wbin("test_kernel", 16, shared_size, &code);
    compile(&wbin, 75).expect("compilation failed")
}

#[test]
fn test_device_load_u32() {
    let ptx = compile_instrs(&single(0x38, 5, 3, 2, 0));
    assert!(ptx.contains("cvt.u64.u32 %rd1, %r3;"), "PTX: {ptx}");
    assert!(ptx.contains("add.u64 %rd1, %rd0, %rd1;"), "PTX: {ptx}");
    assert!(ptx.contains("ld.global.u32 %r5, [%rd1];"), "PTX: {ptx}");
}

#[test]
fn test_device_store_u32() {
    let ptx = compile_instrs(&extended(0x39, 0, 3, 5, 2, 0));
    assert!(ptx.contains("cvt.u64.u32 %rd1, %r3;"), "PTX: {ptx}");
    assert!(ptx.contains("add.u64 %rd1, %rd0, %rd1;"), "PTX: {ptx}");
    assert!(ptx.contains("st.global.u32 [%rd1], %r5;"), "PTX: {ptx}");
}

#[test]
fn test_device_load_u8() {
    let ptx = compile_instrs(&single(0x38, 5, 3, 0, 0));
    assert!(ptx.contains("ld.global.u8 %r5, [%rd1];"), "PTX: {ptx}");
}

#[test]
fn test_shared_load_u32() {
    let ptx = compile_instrs_with_shared(&single(0x30, 5, 3, 2, 0), 4096);
    assert!(ptx.contains("mov.u32 %t0, _shared_mem;"), "PTX: {ptx}");
    assert!(ptx.contains("add.u32 %t0, %t0, %r3;"), "PTX: {ptx}");
    assert!(ptx.contains("ld.shared.u32 %r5, [%t0];"), "PTX: {ptx}");
}

#[test]
fn test_shared_store_u32() {
    let ptx = compile_instrs_with_shared(&extended(0x31, 0, 3, 5, 2, 0), 4096);
    assert!(ptx.contains("st.shared.u32 [%t0], %r5;"), "PTX: {ptx}");
}

#[test]
fn test_device_atomic_add() {
    let ptx = compile_instrs(&extended_scope(0x3D, 5, 3, 4, 0, 2, 0));
    assert!(
        ptx.contains("atom.global.add.u32 %r5, [%rd1], %r4;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_device_atomic_cas() {
    let ptx = compile_instrs(&extended3_scope(0x3D, 5, 3, 4, 6, 8, 2, 0));
    assert!(ptx.contains("atom.global.cas.b32"), "PTX: {ptx}");
}

#[test]
fn test_fence_cta() {
    // fence_acquire modifier = 3 + 8 = 11, scope=1 (CTA) in word1
    let mut code = encode_word0(0x3F, 0, 0, 11, 0).to_le_bytes().to_vec();
    code.extend_from_slice(&1u32.to_le_bytes()); // scope=1 in word1
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("membar.cta;"), "PTX: {ptx}");
}

#[test]
fn test_fence_gl() {
    // fence_acquire modifier = 3 + 8 = 11, scope=2 (Device/GL) in word1
    let mut code = encode_word0(0x3F, 0, 0, 11, 0).to_le_bytes().to_vec();
    code.extend_from_slice(&2u32.to_le_bytes()); // scope=2 in word1
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("membar.gl;"), "PTX: {ptx}");
}
