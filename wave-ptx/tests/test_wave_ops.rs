// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for warp-level intrinsic codegen. Verifies that WAVE wave operations
//!
//! (shuffle, broadcast, ballot, vote, reduce, prefix sum) emit correct PTX shfl.sync,
//! vote.sync, and butterfly reduction sequences with the full-warp 0xFFFFFFFF mask.

use wave_ptx::compile;

const OPCODE_SHIFT: u32 = 24;
const RD_SHIFT: u32 = 16;
const RS1_SHIFT: u32 = 8;
const MODIFIER_SHIFT: u32 = 4;
const EXTENDED_RS2_SHIFT: u32 = 24;

fn encode_word0(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred: u8) -> u32 {
    (u32::from(opcode) << OPCODE_SHIFT) | (u32::from(rd) << RD_SHIFT) | (u32::from(rs1) << RS1_SHIFT) | (u32::from(modifier) << MODIFIER_SHIFT) | u32::from(pred)
}
fn single(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred: u8) -> Vec<u8> {
    encode_word0(opcode, rd, rs1, modifier, pred).to_le_bytes().to_vec()
}
fn extended(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, pred: u8) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, pred).to_le_bytes().to_vec();
    v.extend_from_slice(&(u32::from(rs2) << EXTENDED_RS2_SHIFT).to_le_bytes());
    v
}

fn halt_instruction() -> Vec<u8> {
    single(0x3F, 0, 0, 9, 0)
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
    code.extend_from_slice(&halt_instruction());
    let wbin = build_wbin(&code);
    compile(&wbin, 75).expect("compilation failed")
}

#[test]
fn test_wave_shuffle() {
    let ptx = compile_instrs(&extended(0x3E, 3, 1, 2, 0, 0));
    assert!(
        ptx.contains("shfl.sync.idx.b32 %r3, %r1, %r2, 31, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_shuffle_up() {
    let ptx = compile_instrs(&extended(0x3E, 3, 1, 2, 1, 0));
    assert!(
        ptx.contains("shfl.sync.up.b32 %r3, %r1, %r2, 0, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_shuffle_down() {
    let ptx = compile_instrs(&extended(0x3E, 3, 1, 2, 2, 0));
    assert!(
        ptx.contains("shfl.sync.down.b32 %r3, %r1, %r2, 31, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_shuffle_xor() {
    let ptx = compile_instrs(&extended(0x3E, 3, 1, 2, 3, 0));
    assert!(
        ptx.contains("shfl.sync.bfly.b32 %r3, %r1, %r2, 31, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_broadcast() {
    let ptx = compile_instrs(&extended(0x3E, 3, 1, 2, 4, 0));
    assert!(
        ptx.contains("shfl.sync.idx.b32 %r3, %r1, %r2, 31, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_ballot() {
    let ptx = compile_instrs(&single(0x3E, 3, 1, 5, 0));
    assert!(
        ptx.contains("vote.sync.ballot.b32 %r3, %p1, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_any() {
    let ptx = compile_instrs(&single(0x3E, 2, 1, 6, 0));
    assert!(
        ptx.contains("vote.sync.any.pred %p2, %p1, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_all() {
    let ptx = compile_instrs(&single(0x3E, 2, 1, 7, 0));
    assert!(
        ptx.contains("vote.sync.all.pred %p2, %p1, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_reduce_add() {
    let ptx = compile_instrs(&single(0x3E, 3, 1, 9, 0));
    assert!(
        ptx.contains("shfl.sync.bfly.b32 %t0, %r3, 16, 31, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
    assert!(ptx.contains("add.s32 %r3, %r3, %t0;"), "PTX: {ptx}");
    assert!(
        ptx.contains("shfl.sync.bfly.b32 %t0, %r3, 1, 31, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_wave_reduce_min() {
    let ptx = compile_instrs(&single(0x3E, 3, 1, 10, 0));
    assert!(ptx.contains("min.s32 %r3, %r3, %t0;"), "PTX: {ptx}");
}

#[test]
fn test_wave_reduce_max() {
    let ptx = compile_instrs(&single(0x3E, 3, 1, 11, 0));
    assert!(ptx.contains("max.s32 %r3, %r3, %t0;"), "PTX: {ptx}");
}

#[test]
fn test_wave_prefix_sum() {
    let ptx = compile_instrs(&single(0x3E, 3, 1, 8, 0));
    assert!(
        ptx.contains("shfl.sync.up.b32 %t0|%pt0, %r3, 1, 0, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
    assert!(ptx.contains("@%pt0 add.s32 %r3, %r3, %t0;"), "PTX: {ptx}");
    assert!(
        ptx.contains("shfl.sync.up.b32 %t0|%pt0, %r3, 16, 0, 0xFFFFFFFF;"),
        "PTX: {ptx}"
    );
    assert!(ptx.contains("@!%pt0 mov.b32 %r3, 0;"), "PTX: {ptx}");
}
