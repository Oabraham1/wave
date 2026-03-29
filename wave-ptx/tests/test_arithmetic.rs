// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for integer and floating-point arithmetic codegen. Constructs minimal
//!
//! WBIN files with specific instructions, compiles to PTX, and verifies the output
//! contains expected PTX instruction patterns. Covers add, sub, mul, div, neg, abs,
//! min, max, clamp, and all float unary operations including transcendentals.

use wave_ptx::compile;

const OPCODE_SHIFT: u32 = 26;
const RD_SHIFT: u32 = 21;
const RS1_SHIFT: u32 = 16;
const RS2_SHIFT: u32 = 11;
const MODIFIER_SHIFT: u32 = 7;
const PRED_SHIFT: u32 = 3;
const PRED_NEG_BIT: u32 = 1 << 2;
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

fn encode_predicated(
    opcode: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    modifier: u8,
    flags: u8,
    pred: u8,
    negated: bool,
) -> [u8; 4] {
    let mut word = ((u32::from(opcode) & 0x3F) << OPCODE_SHIFT)
        | ((u32::from(rd) & 0x1F) << RD_SHIFT)
        | ((u32::from(rs1) & 0x1F) << RS1_SHIFT)
        | ((u32::from(rs2) & 0x1F) << RS2_SHIFT)
        | ((u32::from(modifier) & 0x0F) << MODIFIER_SHIFT)
        | ((u32::from(pred) & 0x03) << PRED_SHIFT)
        | (u32::from(flags) & 0x03);
    if negated {
        word |= PRED_NEG_BIT;
    }
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

#[test]
fn test_iadd() {
    let ptx = compile_instrs(&encode(0x00, 5, 3, 4, 0, 0));
    assert!(ptx.contains("add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_isub() {
    let ptx = compile_instrs(&encode(0x01, 2, 0, 1, 0, 0));
    assert!(ptx.contains("sub.s32 %r2, %r0, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_imul() {
    let ptx = compile_instrs(&encode(0x02, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mul.lo.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_imul_hi() {
    let ptx = compile_instrs(&encode(0x03, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mul.hi.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_idiv() {
    let ptx = compile_instrs(&encode(0x05, 3, 1, 2, 0, 0));
    assert!(ptx.contains("div.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_imod() {
    let ptx = compile_instrs(&encode(0x06, 3, 1, 2, 0, 0));
    assert!(ptx.contains("rem.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_ineg() {
    let ptx = compile_instrs(&encode(0x07, 2, 1, 0, 0, 0));
    assert!(ptx.contains("neg.s32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_iabs() {
    let ptx = compile_instrs(&encode(0x08, 2, 1, 0, 0, 0));
    assert!(ptx.contains("abs.s32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_imin() {
    let ptx = compile_instrs(&encode(0x09, 3, 1, 2, 0, 0));
    assert!(ptx.contains("min.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_imax() {
    let ptx = compile_instrs(&encode(0x0A, 3, 1, 2, 0, 0));
    assert!(ptx.contains("max.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_fadd() {
    let ptx = compile_instrs(&encode(0x10, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mov.b32 %f1, %r1;"), "PTX: {ptx}");
    assert!(ptx.contains("mov.b32 %f2, %r2;"), "PTX: {ptx}");
    assert!(ptx.contains("add.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
    assert!(ptx.contains("mov.b32 %r3, %f3;"), "PTX: {ptx}");
}

#[test]
fn test_fsub() {
    let ptx = compile_instrs(&encode(0x11, 3, 1, 2, 0, 0));
    assert!(ptx.contains("sub.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
}

#[test]
fn test_fmul() {
    let ptx = compile_instrs(&encode(0x12, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mul.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
}

#[test]
fn test_fdiv() {
    let ptx = compile_instrs(&encode(0x14, 3, 1, 2, 0, 0));
    assert!(ptx.contains("div.approx.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
}

#[test]
fn test_fsqrt() {
    let ptx = compile_instrs(&encode(0x1A, 2, 1, 0, 0, 0));
    assert!(ptx.contains("sqrt.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fneg() {
    let ptx = compile_instrs(&encode(0x15, 2, 1, 0, 0, 0));
    assert!(ptx.contains("neg.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fabs() {
    let ptx = compile_instrs(&encode(0x16, 2, 1, 0, 0, 0));
    assert!(ptx.contains("abs.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fsin() {
    let ptx = compile_instrs(&encode(0x1B, 2, 1, 0, 8, 0));
    assert!(ptx.contains("sin.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fcos() {
    let ptx = compile_instrs(&encode(0x1B, 2, 1, 0, 9, 0));
    assert!(ptx.contains("cos.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fexp2() {
    let ptx = compile_instrs(&encode(0x1B, 2, 1, 0, 10, 0));
    assert!(ptx.contains("ex2.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_flog2() {
    let ptx = compile_instrs(&encode(0x1B, 2, 1, 0, 11, 0));
    assert!(ptx.contains("lg2.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_ffloor() {
    let ptx = compile_instrs(&encode(0x1B, 2, 1, 0, 2, 0));
    assert!(ptx.contains("cvt.rmi.f32.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_frsqrt() {
    let ptx = compile_instrs(&encode(0x1B, 2, 1, 0, 0, 0));
    assert!(ptx.contains("rsqrt.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_predicated_iadd() {
    let ptx = compile_instrs(&encode_predicated(0x00, 5, 3, 4, 0, 0, 1, false));
    assert!(ptx.contains("@%p1 add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_predicated_negated() {
    let ptx = compile_instrs(&encode_predicated(0x00, 5, 3, 4, 0, 0, 2, true));
    assert!(ptx.contains("@!%p2 add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_mov() {
    let ptx = compile_instrs(&encode(0x3F, 5, 3, 0, 0, MISC_OP_FLAG));
    assert!(ptx.contains("mov.b32 %r5, %r3;"), "PTX: {ptx}");
}

#[test]
fn test_mov_imm() {
    let mut code = encode(0x3F, 5, 0, 0, 1, MISC_OP_FLAG).to_vec();
    code.extend_from_slice(&42u32.to_le_bytes());
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("mov.b32 %r5, 42;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_thread_id() {
    let ptx = compile_instrs(&encode(0x3F, 5, 0, 0, 2, MISC_OP_FLAG));
    assert!(ptx.contains("mov.u32 %r5, %tid.x;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_lane_id() {
    let ptx = compile_instrs(&encode(0x3F, 5, 4, 0, 2, MISC_OP_FLAG));
    assert!(ptx.contains("mov.u32 %r5, %laneid;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_ctaid() {
    let ptx = compile_instrs(&encode(0x3F, 5, 5, 0, 2, MISC_OP_FLAG));
    assert!(ptx.contains("mov.u32 %r5, %ctaid.x;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_wave_width() {
    let ptx = compile_instrs(&encode(0x3F, 5, 14, 0, 2, MISC_OP_FLAG));
    assert!(ptx.contains("mov.u32 %r5, 32;"), "PTX: {ptx}");
}

#[test]
fn test_bitwise_and() {
    let ptx = compile_instrs(&encode(0x20, 3, 1, 2, 0, 0));
    assert!(ptx.contains("and.b32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_bitwise_shl() {
    let ptx = compile_instrs(&encode(0x24, 3, 1, 2, 0, 0));
    assert!(ptx.contains("shl.b32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_bitwise_sar() {
    let ptx = compile_instrs(&encode(0x26, 3, 1, 2, 0, 0));
    assert!(ptx.contains("shr.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_bitcount() {
    let mut code = encode(0x27, 2, 1, 0, 0, 0).to_vec();
    code.extend_from_slice(&[0u8; 4]);
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("popc.b32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_bitrev() {
    let mut code = encode(0x27, 2, 1, 0, 2, 0).to_vec();
    code.extend_from_slice(&[0u8; 4]);
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("brev.b32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_icmp_eq() {
    let ptx = compile_instrs(&encode(0x28, 1, 3, 4, 0, 0));
    assert!(ptx.contains("setp.eq.s32 %p1, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_ucmp_lt() {
    let ptx = compile_instrs(&encode(0x29, 2, 3, 4, 2, 0));
    assert!(ptx.contains("setp.lt.u32 %p2, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_fcmp_gt() {
    let ptx = compile_instrs(&encode(0x2A, 1, 3, 4, 4, 0));
    assert!(ptx.contains("setp.gt.f32 %p1, %f3, %f4;"), "PTX: {ptx}");
}

#[test]
fn test_select() {
    let ptx = compile_instrs(&encode(0x2B, 5, 1, 3, 4, 0));
    assert!(ptx.contains("selp.b32 %r5, %r3, %r4, %p1;"), "PTX: {ptx}");
}

#[test]
fn test_cvt_i32_f32() {
    let ptx = compile_instrs(&encode(0x2C, 2, 1, 0, 2, 0));
    assert!(ptx.contains("cvt.rn.f32.s32 %f2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_kernel_header() {
    let code = halt();
    let wbin = build_wbin("my_kernel", 4, 0, &code);
    let ptx = compile(&wbin, 75).unwrap();
    assert!(ptx.contains(".version 7.5"), "PTX: {ptx}");
    assert!(ptx.contains(".target sm_75"), "PTX: {ptx}");
    assert!(ptx.contains(".address_size 64"), "PTX: {ptx}");
    assert!(ptx.contains(".visible .entry my_kernel("), "PTX: {ptx}");
    assert!(ptx.contains(".param .u64 _device_mem_ptr"), "PTX: {ptx}");
    assert!(ptx.contains(".reg .b32 %r<4>;"), "PTX: {ptx}");
    assert!(ptx.contains(".reg .f32 %f<4>;"), "PTX: {ptx}");
    assert!(ptx.contains(".reg .pred %p<4>;"), "PTX: {ptx}");
    assert!(
        ptx.contains("ld.param.u64 %rd0, [_device_mem_ptr];"),
        "PTX: {ptx}"
    );
    assert!(ptx.contains("ret;"), "PTX: {ptx}");
}

#[test]
fn test_sm_version() {
    let code = halt();
    let wbin = build_wbin("test", 4, 0, &code);
    let ptx = compile(&wbin, 80).unwrap();
    assert!(ptx.contains(".target sm_80"), "PTX: {ptx}");
}

#[test]
fn test_shared_mem_declaration() {
    let code = halt();
    let wbin = build_wbin("test", 4, 4096, &code);
    let ptx = compile(&wbin, 75).unwrap();
    assert!(
        ptx.contains(".shared .align 4 .b8 _shared_mem[4096];"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_no_shared_mem_when_zero() {
    let code = halt();
    let wbin = build_wbin("test", 4, 0, &code);
    let ptx = compile(&wbin, 75).unwrap();
    assert!(!ptx.contains("_shared_mem"), "PTX: {ptx}");
}

#[test]
fn test_halt_emits_ret() {
    let ptx = compile_instrs(&[]);
    assert!(ptx.contains("ret;"), "should contain ret");
}

#[test]
fn test_barrier() {
    let ptx = compile_instrs(&encode(0x3F, 0, 0, 0, 2, SYNC_OP_FLAG));
    assert!(ptx.contains("bar.sync 0;"), "PTX: {ptx}");
}
