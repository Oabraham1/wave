// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for full WAVE program compilation. Builds complete WBIN files
//!
//! containing multi-instruction programs, compiles them to MSL, and verifies the
//! output is structurally correct with proper kernel signatures, register declarations,
//! instruction translations, and control flow nesting.

use wave_metal::compile;

const OPCODE_SHIFT: u32 = 24;
const RD_SHIFT: u32 = 16;
const RS1_SHIFT: u32 = 8;
const MODIFIER_SHIFT: u32 = 4;
const EXTENDED_RS2_SHIFT: u32 = 24;
const SYNC_MODIFIER_OFFSET: u8 = 8;

fn encode_word0(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred_reg: u8, pred_neg: bool) -> u32 {
    (u32::from(opcode) << OPCODE_SHIFT)
        | (u32::from(rd) << RD_SHIFT)
        | (u32::from(rs1) << RS1_SHIFT)
        | (u32::from(modifier) << MODIFIER_SHIFT)
        | u32::from(pred_reg & 0x03)
        | if pred_neg { 1 << 2 } else { 0 }
}

fn encode_ext(rs2: u8) -> u32 {
    u32::from(rs2) << EXTENDED_RS2_SHIFT
}

/// Single-word instruction (4 bytes)
fn single(opcode: u8, rd: u8, rs1: u8, modifier: u8) -> Vec<u8> {
    encode_word0(opcode, rd, rs1, modifier, 0, false)
        .to_le_bytes()
        .to_vec()
}

/// Extended instruction (8 bytes: word0 + word1 with rs2)
fn extended(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, 0, false)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(&encode_ext(rs2).to_le_bytes());
    v
}

/// Halt: Control opcode 0x3F, modifier = Halt(1) + SYNC_MODIFIER_OFFSET(8) = 9
/// Halt does NOT read word1
fn halt() -> Vec<u8> {
    single(0x3F, 0, 0, 1 + SYNC_MODIFIER_OFFSET)
}

/// MovSr: Misc opcode 0x41, modifier=2, rs1=sr_index
/// MovSr does NOT read word1
fn mov_sr(rd: u8, sr: u8) -> Vec<u8> {
    single(0x41, rd, sr, 2)
}

/// MovImm: Misc opcode 0x41, modifier=1, word1=immediate
fn mov_imm(rd: u8, imm: u32) -> Vec<u8> {
    let mut code = encode_word0(0x41, rd, 0, 1, 0, false)
        .to_le_bytes()
        .to_vec();
    code.extend_from_slice(&imm.to_le_bytes());
    code
}

fn build_wbin(kernel_name: &str, reg_count: u32, local_mem: u32, code: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    let name_bytes: Vec<u8> = kernel_name.bytes().chain(std::iter::once(0)).collect();

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

#[test]
fn test_vector_add_program() {
    let mut code = Vec::new();

    code.extend_from_slice(&mov_sr(0, 0));
    code.extend_from_slice(&mov_sr(1, 5));
    code.extend_from_slice(&mov_sr(2, 8));
    code.extend_from_slice(&extended(0x02, 3, 1, 2, 0));
    code.extend_from_slice(&extended(0x00, 4, 3, 0, 0));
    code.extend_from_slice(&mov_imm(5, 4));
    code.extend_from_slice(&extended(0x02, 6, 4, 5, 0));
    code.extend_from_slice(&single(0x38, 7, 6, 2));
    code.extend_from_slice(&mov_imm(8, 1024));
    code.extend_from_slice(&extended(0x00, 9, 6, 8, 0));
    code.extend_from_slice(&single(0x38, 10, 9, 2));
    code.extend_from_slice(&extended(0x10, 11, 7, 10, 0));
    code.extend_from_slice(&mov_imm(12, 2048));
    code.extend_from_slice(&extended(0x00, 13, 6, 12, 0));
    code.extend_from_slice(&extended(0x39, 0, 13, 11, 2));
    code.extend_from_slice(&halt());

    let wbin = build_wbin("vector_add", 16, 0, &code);
    let msl = compile(&wbin).unwrap();

    assert!(msl.contains("kernel void vector_add("), "MSL: {msl}");
    assert!(msl.contains("r0 = (uint32_t)tid.x;"), "MSL: {msl}");
    assert!(msl.contains("r1 = (uint32_t)gid.x;"), "MSL: {msl}");
    assert!(msl.contains("r2 = (uint32_t)tsize.x;"), "MSL: {msl}");
    assert!(msl.contains("r3 = r1 * r2;"), "MSL: {msl}");
    assert!(msl.contains("r4 = r3 + r0;"), "MSL: {msl}");
    assert!(msl.contains("r5 = 4u;"), "MSL: {msl}");
    assert!(msl.contains("r6 = r4 * r5;"), "MSL: {msl}");
    assert!(
        msl.contains("r7 = (uint32_t)(*(device uint32_t*)(device_mem + r6));"),
        "MSL: {msl}"
    );
    assert!(msl.contains("r8 = 1024u;"), "MSL: {msl}");
    assert!(msl.contains("r11 = ri(rf(r7) + rf(r10));"), "MSL: {msl}");
    assert!(
        msl.contains("*(device uint32_t*)(device_mem + r13) = (uint32_t)r11;"),
        "MSL: {msl}"
    );
    assert!(msl.contains("return;"), "MSL: {msl}");
}

#[test]
fn test_reduction_with_barrier() {
    let mut code = Vec::new();

    code.extend_from_slice(&mov_sr(0, 0));
    code.extend_from_slice(&mov_imm(1, 4));
    code.extend_from_slice(&extended(0x02, 2, 0, 1, 0));
    code.extend_from_slice(&single(0x38, 3, 2, 2));
    code.extend_from_slice(&extended(0x31, 0, 2, 3, 2));
    // Barrier = SyncOp::Barrier(2) + SYNC_MODIFIER_OFFSET(8) = 10, does NOT read word1
    code.extend_from_slice(&single(0x3F, 0, 0, 2 + SYNC_MODIFIER_OFFSET));
    code.extend_from_slice(&halt());

    let wbin = build_wbin("reduction", 8, 4096, &code);
    let msl = compile(&wbin).unwrap();

    assert!(msl.contains("kernel void reduction("), "MSL: {msl}");
    assert!(
        msl.contains("threadgroup uint8_t local_mem[4096];"),
        "MSL: {msl}"
    );
    assert!(
        msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"),
        "MSL: {msl}"
    );
}

#[test]
fn test_conditional_program() {
    let mut code = Vec::new();

    code.extend_from_slice(&mov_sr(0, 0));
    code.extend_from_slice(&mov_imm(1, 16));
    code.extend_from_slice(&extended(0x29, 1, 0, 1, 2));

    // Control::If = modifier 0, rs1=1 - does NOT read word1
    code.extend_from_slice(&single(0x3F, 0, 1, 0));
    code.extend_from_slice(&mov_imm(2, 1));
    // Control::Else = modifier 1 - does NOT read word1
    code.extend_from_slice(&single(0x3F, 0, 0, 1));
    code.extend_from_slice(&mov_imm(2, 0));
    // Control::Endif = modifier 2 - does NOT read word1
    code.extend_from_slice(&single(0x3F, 0, 0, 2));

    code.extend_from_slice(&halt());

    let wbin = build_wbin("conditional", 8, 0, &code);
    let msl = compile(&wbin).unwrap();

    assert!(msl.contains("p1 = r0 < r1;"), "MSL: {msl}");
    assert!(msl.contains("if (p1) {"), "MSL: {msl}");
    assert!(msl.contains("} else {"), "MSL: {msl}");
}

#[test]
fn test_empty_kernel_name_uses_default() {
    let code = halt();
    let wbin = build_wbin("", 4, 0, &code);
    let msl = compile(&wbin).unwrap();
    assert!(msl.contains("kernel void wave_kernel("), "MSL: {msl}");
}

#[test]
fn test_no_local_mem_omits_declaration() {
    let code = halt();
    let wbin = build_wbin("test", 4, 0, &code);
    let msl = compile(&wbin).unwrap();
    assert!(!msl.contains("threadgroup uint8_t local_mem"), "MSL: {msl}");
}

#[test]
fn test_helper_functions_present() {
    let code = halt();
    let wbin = build_wbin("test", 4, 0, &code);
    let msl = compile(&wbin).unwrap();
    assert!(
        msl.contains("inline float rf(uint32_t r) { return as_type<float>(r); }"),
        "MSL: {msl}"
    );
    assert!(
        msl.contains("inline uint32_t ri(float f) { return as_type<uint32_t>(f); }"),
        "MSL: {msl}"
    );
    assert!(msl.contains("inline half rh(uint32_t r)"), "MSL: {msl}");
    assert!(msl.contains("inline half2 rh2(uint32_t r)"), "MSL: {msl}");
}

#[test]
fn test_wave_count_declaration() {
    let code = halt();
    let wbin = build_wbin("test", 4, 0, &code);
    let msl = compile(&wbin).unwrap();
    assert!(
        msl.contains("uint wave_count = (tsize.x * tsize.y * tsize.z + 31u) / 32u;"),
        "MSL: {msl}"
    );
}

#[test]
fn test_invalid_wbin_magic() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(b"NOPE");
    let result = compile(&data);
    assert!(result.is_err());
}
