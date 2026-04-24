// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for full WAVE program compilation to SYCL. Builds complete WBIN
//!
//! files, compiles to SYCL C++, and verifies structural correctness including the
//! launch function wrapper, nd_range dispatch, sub-group accessor bindings, and
//! SYCL-specific instruction mappings.

use wave_sycl::compile;

const OPCODE_SHIFT: u32 = 24;
const RD_SHIFT: u32 = 16;
const RS1_SHIFT: u32 = 8;
const MODIFIER_SHIFT: u32 = 4;
const EXTENDED_RS2_SHIFT: u32 = 24;
const SYNC_MODIFIER_OFFSET: u8 = 8;

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
fn halt_instruction() -> Vec<u8> {
    single(0x3F, 0, 0, SYNC_MODIFIER_OFFSET + 1, 0)
}

fn mov_sr(rd: u8, sr: u8) -> Vec<u8> {
    single(0x41, rd, sr, 2, 0)
}

fn mov_imm(rd: u8, imm: u32) -> Vec<u8> {
    let mut v = encode_word0(0x41, rd, 0, 1, 0).to_le_bytes().to_vec();
    v.extend_from_slice(&imm.to_le_bytes());
    v
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

#[test]
fn test_vector_add_program() {
    let mut code = Vec::new();
    code.extend_from_slice(&mov_sr(0, 0));
    code.extend_from_slice(&mov_sr(1, 5));
    code.extend_from_slice(&mov_sr(2, 8));
    code.extend_from_slice(&extended(0x02, 3, 1, 2, 0, 0));
    code.extend_from_slice(&extended(0x00, 4, 3, 0, 0, 0));
    code.extend_from_slice(&mov_imm(5, 4));
    code.extend_from_slice(&extended(0x02, 6, 4, 5, 0, 0));
    code.extend_from_slice(&single(0x38, 7, 6, 2, 0));
    code.extend_from_slice(&mov_imm(8, 1024));
    code.extend_from_slice(&extended(0x00, 9, 6, 8, 0, 0));
    code.extend_from_slice(&single(0x38, 10, 9, 2, 0));
    code.extend_from_slice(&extended(0x10, 11, 7, 10, 0, 0));
    code.extend_from_slice(&mov_imm(12, 2048));
    code.extend_from_slice(&extended(0x00, 13, 6, 12, 0, 0));
    code.extend_from_slice(&extended(0x39, 0, 13, 11, 2, 0));
    code.extend_from_slice(&halt_instruction());

    let wbin = build_wbin("vector_add", 16, 0, &code);
    let s = compile(&wbin).unwrap();

    assert!(s.contains("void vector_add_launch(queue& q"), "SYCL: {s}");
    assert!(
        s.contains("r0 = (uint32_t)it.get_local_id(0);"),
        "SYCL: {s}"
    );
    assert!(s.contains("r1 = (uint32_t)it.get_group(0);"), "SYCL: {s}");
    assert!(
        s.contains("r2 = (uint32_t)it.get_local_range(0);"),
        "SYCL: {s}"
    );
    assert!(s.contains("r3 = r1 * r2;"), "SYCL: {s}");
    assert!(s.contains("r11 = ri(rf(r7) + rf(r10));"), "SYCL: {s}");
    assert!(
        s.contains("*(uint32_t*)(device_mem_usm + r13) = (uint32_t)r11;"),
        "SYCL: {s}"
    );
}

#[test]
fn test_reduction_with_barrier() {
    let mut code = Vec::new();
    code.extend_from_slice(&mov_sr(0, 0));
    code.extend_from_slice(&mov_imm(1, 4));
    code.extend_from_slice(&extended(0x02, 2, 0, 1, 0, 0));
    code.extend_from_slice(&single(0x38, 3, 2, 2, 0));
    code.extend_from_slice(&extended(0x31, 0, 2, 3, 2, 0));
    code.extend_from_slice(&single(0x3F, 0, 0, SYNC_MODIFIER_OFFSET + 2, 0));
    code.extend_from_slice(&halt_instruction());

    let wbin = build_wbin("reduction", 8, 4096, &code);
    let s = compile(&wbin).unwrap();

    assert!(s.contains("void reduction_launch(queue& q"), "SYCL: {s}");
    assert!(s.contains("local_accessor<uint8_t, 1>"), "SYCL: {s}");
    assert!(s.contains("group_barrier(it.get_group());"), "SYCL: {s}");
}

#[test]
fn test_empty_kernel_name() {
    let code = halt_instruction();
    let wbin = build_wbin("", 4, 0, &code);
    let s = compile(&wbin).unwrap();
    assert!(s.contains("void wave_kernel_launch(queue& q"), "SYCL: {s}");
}

#[test]
fn test_invalid_wbin_magic() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(b"NOPE");
    let result = compile(&data);
    assert!(result.is_err());
}

#[test]
fn test_sub_group_width_not_hardcoded() {
    let code = halt_instruction();
    let wbin = build_wbin("test", 4, 0, &code);
    let s = compile(&wbin).unwrap();
    assert!(
        !s.contains("warpSize"),
        "should use sub_group accessors, not warpSize: {s}"
    );
    assert!(s.contains("get_sub_group()"), "SYCL: {s}");
}
