// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for full WAVE program compilation to HIP. Builds complete WBIN
//!
//! files containing multi-instruction programs, compiles them to HIP C++, and verifies
//! the output is structurally correct with proper __global__ declarations, register
//! declarations, instruction translations, and control flow.

use wave_hip::compile;

const OPCODE_SHIFT: u32 = 26;
const RD_SHIFT: u32 = 21;
const RS1_SHIFT: u32 = 16;
const RS2_SHIFT: u32 = 11;
const MODIFIER_SHIFT: u32 = 7;
const SYNC_OP_FLAG: u8 = 0x01;
const MISC_OP_FLAG: u8 = 0x02;

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

fn mov_sr(rd: u8, sr: u8) -> [u8; 4] {
    encode(0x3F, rd, sr, 0, 2, MISC_OP_FLAG)
}

fn mov_imm(rd: u8, imm: u32) -> Vec<u8> {
    let mut code = encode(0x3F, rd, 0, 0, 1, MISC_OP_FLAG).to_vec();
    code.extend_from_slice(&imm.to_le_bytes());
    code
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
    code.extend_from_slice(&encode(0x02, 3, 1, 2, 0, 0));
    code.extend_from_slice(&encode(0x00, 4, 3, 0, 0, 0));
    code.extend_from_slice(&mov_imm(5, 4));
    code.extend_from_slice(&encode(0x02, 6, 4, 5, 0, 0));
    code.extend_from_slice(&encode(0x38, 7, 6, 0, 2, 0));
    code.extend_from_slice(&mov_imm(8, 1024));
    code.extend_from_slice(&encode(0x00, 9, 6, 8, 0, 0));
    code.extend_from_slice(&encode(0x38, 10, 9, 0, 2, 0));
    code.extend_from_slice(&encode(0x10, 11, 7, 10, 0, 0));
    code.extend_from_slice(&mov_imm(12, 2048));
    code.extend_from_slice(&encode(0x00, 13, 6, 12, 0, 0));
    code.extend_from_slice(&encode(0x39, 0, 13, 11, 2, 0));
    code.extend_from_slice(&halt());

    let wbin = build_wbin("vector_add", 16, 0, &code);
    let hip = compile(&wbin).unwrap();

    assert!(hip.contains("__global__ void vector_add("), "HIP: {hip}");
    assert!(hip.contains("r0 = (uint32_t)threadIdx.x;"), "HIP: {hip}");
    assert!(hip.contains("r1 = (uint32_t)blockIdx.x;"), "HIP: {hip}");
    assert!(hip.contains("r2 = (uint32_t)blockDim.x;"), "HIP: {hip}");
    assert!(hip.contains("r3 = r1 * r2;"), "HIP: {hip}");
    assert!(hip.contains("r4 = r3 + r0;"), "HIP: {hip}");
    assert!(hip.contains("r11 = ri(rf(r7) + rf(r10));"), "HIP: {hip}");
    assert!(hip.contains("*(uint32_t*)(device_mem + r13) = (uint32_t)r11;"), "HIP: {hip}");
}

#[test]
fn test_reduction_with_barrier() {
    let mut code = Vec::new();
    code.extend_from_slice(&mov_sr(0, 0));
    code.extend_from_slice(&mov_imm(1, 4));
    code.extend_from_slice(&encode(0x02, 2, 0, 1, 0, 0));
    code.extend_from_slice(&encode(0x38, 3, 2, 0, 2, 0));
    code.extend_from_slice(&encode(0x31, 0, 2, 3, 2, 0));
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 2, SYNC_OP_FLAG));
    code.extend_from_slice(&halt());

    let wbin = build_wbin("reduction", 8, 4096, &code);
    let hip = compile(&wbin).unwrap();

    assert!(hip.contains("__global__ void reduction("), "HIP: {hip}");
    assert!(hip.contains("extern __shared__ uint8_t local_mem[];"), "HIP: {hip}");
    assert!(hip.contains("__syncthreads();"), "HIP: {hip}");
}

#[test]
fn test_empty_kernel_name() {
    let code = halt();
    let wbin = build_wbin("", 4, 0, &code);
    let hip = compile(&wbin).unwrap();
    assert!(hip.contains("__global__ void wave_kernel("), "HIP: {hip}");
}

#[test]
fn test_invalid_wbin_magic() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(b"NOPE");
    let result = compile(&data);
    assert!(result.is_err());
}

#[test]
fn test_warp_size_used_not_hardcoded() {
    let code = halt();
    let wbin = build_wbin("test", 4, 0, &code);
    let hip = compile(&wbin).unwrap();
    assert!(hip.contains("warpSize"), "HIP: {hip}");
    let wave_count_line = hip.lines().find(|l| l.contains("wave_count")).unwrap();
    assert!(wave_count_line.contains("warpSize"), "wave_count should use warpSize: {wave_count_line}");
}
