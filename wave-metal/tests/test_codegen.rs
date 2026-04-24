// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for individual instruction codegen. Constructs minimal WBIN files with
//!
//! specific WAVE instructions, compiles them to MSL, and verifies the output contains
//! the expected Metal code fragments. Tests cover integer arithmetic, float operations,
//! bitwise ops, memory access, control flow, special registers, and wave intrinsics.

use wave_metal::compile;

const OPCODE_SHIFT: u32 = 24;
const RD_SHIFT: u32 = 16;
const RS1_SHIFT: u32 = 8;
const MODIFIER_SHIFT: u32 = 4;
const EXTENDED_RS2_SHIFT: u32 = 24;
const EXTENDED_RS3_SHIFT: u32 = 16;
const EXTENDED_SCOPE_SHIFT: u32 = 0;
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

fn encode_ext3(rs2: u8, rs3: u8) -> u32 {
    (u32::from(rs2) << EXTENDED_RS2_SHIFT) | (u32::from(rs3) << EXTENDED_RS3_SHIFT)
}

fn encode_ext_scope(rs2: u8, scope: u8) -> u32 {
    (u32::from(rs2) << EXTENDED_RS2_SHIFT) | (u32::from(scope) << EXTENDED_SCOPE_SHIFT)
}

/// Single-word instruction (4 bytes) - no predicate
fn single(opcode: u8, rd: u8, rs1: u8, modifier: u8) -> Vec<u8> {
    encode_word0(opcode, rd, rs1, modifier, 0, false)
        .to_le_bytes()
        .to_vec()
}

/// Extended instruction (8 bytes: word0 + word1 with rs2) - no predicate
fn extended(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, 0, false)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(&encode_ext(rs2).to_le_bytes());
    v
}

/// Extended instruction with scope in word1 bits 0-1
fn extended_scope(
    opcode: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    modifier: u8,
    scope: u8,
) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, 0, false)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(&encode_ext_scope(rs2, scope).to_le_bytes());
    v
}

/// Halt instruction: Control opcode 0x3F, modifier = Halt(1) + SYNC_MODIFIER_OFFSET(8) = 9
/// Halt does NOT read word1 in the decoder (single-word sync op)
fn halt_instruction() -> Vec<u8> {
    single(0x3F, 0, 0, 1 + SYNC_MODIFIER_OFFSET)
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

fn compile_instructions(instructions: &[u8]) -> String {
    let mut code = instructions.to_vec();
    code.extend_from_slice(&halt_instruction());
    let wbin = build_wbin("test_kernel", 16, 0, &code);
    compile(&wbin).expect("compilation failed")
}

fn compile_instructions_with_local_mem(instructions: &[u8], local_mem: u32) -> String {
    let mut code = instructions.to_vec();
    code.extend_from_slice(&halt_instruction());
    let wbin = build_wbin("test_kernel", 16, local_mem, &code);
    compile(&wbin).expect("compilation failed")
}

#[test]
fn test_iadd() {
    let code = extended(0x00, 5, 3, 4, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = r3 + r4;"), "MSL: {msl}");
}

#[test]
fn test_isub() {
    let code = extended(0x01, 2, 0, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = r0 - r1;"), "MSL: {msl}");
}

#[test]
fn test_imul() {
    let code = extended(0x02, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = r1 * r2;"), "MSL: {msl}");
}

#[test]
fn test_imul_hi() {
    let code = extended(0x03, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("(uint32_t)(((uint64_t)r1 * (uint64_t)r2) >> 32)"),
        "MSL: {msl}"
    );
}

#[test]
fn test_idiv() {
    let code = extended(0x05, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("(uint32_t)((int32_t)r1 / (int32_t)r2)"),
        "MSL: {msl}"
    );
}

#[test]
fn test_imod() {
    let code = extended(0x06, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("(uint32_t)((int32_t)r1 % (int32_t)r2)"),
        "MSL: {msl}"
    );
}

#[test]
fn test_ineg() {
    let code = single(0x07, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("(uint32_t)(-(int32_t)r1)"), "MSL: {msl}");
}

#[test]
fn test_iabs() {
    let code = single(0x08, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("(uint32_t)abs((int32_t)r1)"), "MSL: {msl}");
}

#[test]
fn test_imin() {
    let code = extended(0x09, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("(uint32_t)min((int32_t)r1, (int32_t)r2)"),
        "MSL: {msl}"
    );
}

#[test]
fn test_imax() {
    let code = extended(0x0A, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("(uint32_t)max((int32_t)r1, (int32_t)r2)"),
        "MSL: {msl}"
    );
}

#[test]
fn test_fadd() {
    let code = extended(0x10, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = ri(rf(r1) + rf(r2));"), "MSL: {msl}");
}

#[test]
fn test_fsub() {
    let code = extended(0x11, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = ri(rf(r1) - rf(r2));"), "MSL: {msl}");
}

#[test]
fn test_fmul() {
    let code = extended(0x12, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = ri(rf(r1) * rf(r2));"), "MSL: {msl}");
}

#[test]
fn test_fdiv() {
    let code = extended(0x14, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = ri(rf(r1) / rf(r2));"), "MSL: {msl}");
}

#[test]
fn test_fsqrt() {
    let code = single(0x1A, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(sqrt(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_fneg() {
    let code = single(0x15, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(-rf(r1));"), "MSL: {msl}");
}

#[test]
fn test_fabs() {
    let code = single(0x16, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(abs(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_fsin() {
    let code = single(0x1B, 2, 1, 8);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(sin(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_fcos() {
    let code = single(0x1B, 2, 1, 9);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(cos(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_fexp2() {
    let code = single(0x1B, 2, 1, 10);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(exp2(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_flog2() {
    let code = single(0x1B, 2, 1, 11);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(log2(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_ftrunc() {
    let code = single(0x1B, 2, 1, 5);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(trunc(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_ffract() {
    let code = single(0x1B, 2, 1, 6);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(fract(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_ffloor() {
    let code = single(0x1B, 2, 1, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(floor(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_funary_frsqrt() {
    let code = single(0x1B, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri(rsqrt(rf(r1)));"), "MSL: {msl}");
}

#[test]
fn test_and() {
    let code = extended(0x20, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = r1 & r2;"), "MSL: {msl}");
}

#[test]
fn test_or() {
    let code = extended(0x21, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = r1 | r2;"), "MSL: {msl}");
}

#[test]
fn test_xor() {
    let code = extended(0x22, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = r1 ^ r2;"), "MSL: {msl}");
}

#[test]
fn test_not() {
    let code = single(0x23, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ~r1;"), "MSL: {msl}");
}

#[test]
fn test_shl() {
    let code = extended(0x24, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = r1 << (r2 & 0x1Fu);"), "MSL: {msl}");
}

#[test]
fn test_shr() {
    let code = extended(0x25, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = r1 >> (r2 & 0x1Fu);"), "MSL: {msl}");
}

#[test]
fn test_sar() {
    let code = extended(0x26, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("(uint32_t)((int32_t)r1 >> (r2 & 0x1Fu))"),
        "MSL: {msl}"
    );
}

#[test]
fn test_bitcount() {
    let code = single(0x27, 2, 1, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = popcount(r1);"), "MSL: {msl}");
}

#[test]
fn test_bitfind() {
    let code = single(0x27, 2, 1, 1);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("(r1 == 0u) ? 0xFFFFFFFFu : ctz(r1)"),
        "MSL: {msl}"
    );
}

#[test]
fn test_bitrev() {
    let code = single(0x27, 2, 1, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = reverse_bits(r1);"), "MSL: {msl}");
}

#[test]
fn test_mov() {
    // Misc opcode 0x41, modifier=0 (Mov) - Mov does NOT read word1
    let code = single(0x41, 5, 3, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = r3;"), "MSL: {msl}");
}

#[test]
fn test_mov_imm() {
    // Misc opcode 0x41, modifier=1 (MovImm) - reads word1 as immediate
    let mut code = encode_word0(0x41, 5, 0, 1, 0, false)
        .to_le_bytes()
        .to_vec();
    code.extend_from_slice(&42u32.to_le_bytes());
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = 42u;"), "MSL: {msl}");
}

#[test]
fn test_mov_sr_thread_id_x() {
    // Misc opcode 0x41, modifier=2 (MovSr) - does NOT read word1
    let code = single(0x41, 5, 0, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = (uint32_t)tid.x;"), "MSL: {msl}");
}

#[test]
fn test_mov_sr_lane_id() {
    let code = single(0x41, 5, 4, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = (uint32_t)lane_id;"), "MSL: {msl}");
}

#[test]
fn test_mov_sr_workgroup_id() {
    let code = single(0x41, 5, 5, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = (uint32_t)gid.x;"), "MSL: {msl}");
}

#[test]
fn test_mov_sr_wave_width() {
    let code = single(0x41, 5, 14, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = (uint32_t)32u;"), "MSL: {msl}");
}

#[test]
fn test_device_load_u32() {
    let code = single(0x38, 5, 3, 2);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("r5 = (uint32_t)(*(device uint32_t*)(device_mem + r3));"),
        "MSL: {msl}"
    );
}

#[test]
fn test_device_store_u32() {
    let code = extended(0x39, 0, 3, 5, 2);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("*(device uint32_t*)(device_mem + r3) = (uint32_t)r5;"),
        "MSL: {msl}"
    );
}

#[test]
fn test_local_load_u32() {
    let code = single(0x30, 5, 3, 2);
    let msl = compile_instructions_with_local_mem(&code, 4096);
    assert!(
        msl.contains("r5 = (uint32_t)(*(threadgroup uint32_t*)(local_mem + r3));"),
        "MSL: {msl}"
    );
    assert!(
        msl.contains("threadgroup uint8_t local_mem[4096];"),
        "MSL: {msl}"
    );
}

#[test]
fn test_local_store_u32() {
    let code = extended(0x31, 0, 3, 5, 2);
    let msl = compile_instructions_with_local_mem(&code, 4096);
    assert!(
        msl.contains("*(threadgroup uint32_t*)(local_mem + r3) = (uint32_t)r5;"),
        "MSL: {msl}"
    );
}

#[test]
fn test_device_load_u8() {
    let code = single(0x38, 5, 3, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("r5 = (uint32_t)(*(device uint8_t*)(device_mem + r3));"),
        "MSL: {msl}"
    );
}

#[test]
fn test_barrier() {
    // Barrier = SyncOp::Barrier(2) + SYNC_MODIFIER_OFFSET(8) = 10
    // Barrier does NOT read word1
    let code = single(0x3F, 0, 0, 2 + SYNC_MODIFIER_OFFSET);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"),
        "MSL: {msl}"
    );
}

#[test]
fn test_halt() {
    // Halt = SyncOp::Halt(1) + SYNC_MODIFIER_OFFSET(8) = 9
    let code = single(0x3F, 0, 0, 1 + SYNC_MODIFIER_OFFSET);
    let msl = compile_instructions(&code);
    assert!(msl.contains("return;"), "MSL: {msl}");
}

#[test]
fn test_if_else_endif() {
    let mut code = Vec::new();
    // Control::If = modifier 0, rs1=1 (predicate register) - does NOT read word1
    code.extend_from_slice(&single(0x3F, 0, 1, 0));
    code.extend_from_slice(&extended(0x00, 5, 3, 4, 0));
    // Control::Else = modifier 1 - does NOT read word1
    code.extend_from_slice(&single(0x3F, 0, 0, 1));
    code.extend_from_slice(&extended(0x00, 5, 1, 2, 0));
    // Control::Endif = modifier 2 - does NOT read word1
    code.extend_from_slice(&single(0x3F, 0, 0, 2));
    let msl = compile_instructions(&code);
    assert!(msl.contains("if (p1) {"), "MSL: {msl}");
    assert!(msl.contains("} else {"), "MSL: {msl}");
    assert!(msl.contains("r5 = r3 + r4;"), "MSL: {msl}");
    assert!(msl.contains("r5 = r1 + r2;"), "MSL: {msl}");
}

#[test]
fn test_loop_break() {
    let mut code = Vec::new();
    // Control::Loop = modifier 3
    code.extend_from_slice(&single(0x3F, 0, 0, 3));
    // Control::Break = modifier 4, rs1=1
    code.extend_from_slice(&single(0x3F, 0, 1, 4));
    // Control::Endloop = modifier 6
    code.extend_from_slice(&single(0x3F, 0, 0, 6));
    let msl = compile_instructions(&code);
    assert!(msl.contains("while (true) {"), "MSL: {msl}");
    assert!(msl.contains("if (p1) break;"), "MSL: {msl}");
}

#[test]
fn test_predicated_instruction() {
    let mut code = Vec::new();
    // Control::If = modifier 0, rs1=1
    code.extend_from_slice(&single(0x3F, 0, 1, 0));
    code.extend_from_slice(&extended(0x00, 5, 3, 4, 0));
    // Control::Endif = modifier 2
    code.extend_from_slice(&single(0x3F, 0, 0, 2));
    let msl = compile_instructions(&code);
    assert!(msl.contains("if (p1) {"), "MSL: {msl}");
    assert!(msl.contains("r5 = r3 + r4;"), "MSL: {msl}");
}

#[test]
fn test_predicated_negated() {
    let mut code = Vec::new();
    // Control::If = modifier 0, rs1=2
    code.extend_from_slice(&single(0x3F, 0, 2, 0));
    code.extend_from_slice(&extended(0x00, 5, 3, 4, 0));
    // Control::Endif = modifier 2
    code.extend_from_slice(&single(0x3F, 0, 0, 2));
    let msl = compile_instructions(&code);
    assert!(msl.contains("if (p2) {"), "MSL: {msl}");
    assert!(msl.contains("r5 = r3 + r4;"), "MSL: {msl}");
}

#[test]
fn test_icmp_eq() {
    let code = extended(0x28, 1, 3, 4, 0);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("p1 = (int32_t)r3 == (int32_t)r4;"),
        "MSL: {msl}"
    );
}

#[test]
fn test_ucmp_lt() {
    let code = extended(0x29, 2, 3, 4, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("p2 = r3 < r4;"), "MSL: {msl}");
}

#[test]
fn test_fcmp_gt() {
    let code = extended(0x2A, 1, 3, 4, 4);
    let msl = compile_instructions(&code);
    assert!(msl.contains("p1 = rf(r3) > rf(r4);"), "MSL: {msl}");
}

#[test]
fn test_select() {
    let mut code = encode_word0(0x2B, 5, 1, 0, 0, false).to_le_bytes().to_vec();
    code.extend_from_slice(&encode_ext3(3, 4).to_le_bytes());
    let msl = compile_instructions(&code);
    assert!(msl.contains("r5 = p1 ? r3 : r4;"), "MSL: {msl}");
}

#[test]
fn test_cvt_i32_f32() {
    let code = single(0x2C, 2, 1, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = ri((float)((int32_t)r1));"), "MSL: {msl}");
}

#[test]
fn test_cvt_f32_u32() {
    let code = single(0x2C, 2, 1, 1);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r2 = (uint32_t)rf(r1);"), "MSL: {msl}");
}

#[test]
fn test_wave_shuffle() {
    let code = extended(0x3E, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = simd_shuffle(r1, r2);"), "MSL: {msl}");
}

#[test]
fn test_wave_broadcast() {
    let code = extended(0x3E, 3, 1, 2, 4);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = simd_broadcast(r1, r2);"), "MSL: {msl}");
}

#[test]
fn test_wave_reduce_add() {
    let code = single(0x3E, 3, 1, 9);
    let msl = compile_instructions(&code);
    assert!(msl.contains("r3 = simd_sum(r1);"), "MSL: {msl}");
}

#[test]
fn test_wave_ballot() {
    let code = single(0x3E, 3, 1, 5);
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("r3 = (uint32_t)simd_ballot(p1);"),
        "MSL: {msl}"
    );
}

#[test]
fn test_wave_any() {
    let code = single(0x3E, 2, 1, 6);
    let msl = compile_instructions(&code);
    assert!(msl.contains("p2 = simd_any(p1);"), "MSL: {msl}");
}

#[test]
fn test_kernel_header_structure() {
    let code = halt_instruction();
    let wbin = build_wbin("my_kernel", 4, 0, &code);
    let msl = compile(&wbin).unwrap();
    assert!(msl.contains("#include <metal_stdlib>"), "MSL: {msl}");
    assert!(msl.contains("using namespace metal;"), "MSL: {msl}");
    assert!(msl.contains("kernel void my_kernel("), "MSL: {msl}");
    assert!(
        msl.contains("device uint8_t* device_mem [[buffer(0)]]"),
        "MSL: {msl}"
    );
    assert!(
        msl.contains("uint3 tid [[thread_position_in_threadgroup]]"),
        "MSL: {msl}"
    );
    assert!(
        msl.contains("uint lane_id [[thread_index_in_simdgroup]]"),
        "MSL: {msl}"
    );
    assert!(msl.contains("inline float rf(uint32_t r)"), "MSL: {msl}");
    assert!(msl.contains("inline uint32_t ri(float f)"), "MSL: {msl}");
}

#[test]
fn test_register_declarations() {
    let code = halt_instruction();
    let wbin = build_wbin("test_kernel", 4, 1024, &code);
    let msl = compile(&wbin).unwrap();
    assert!(
        msl.contains("uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;"),
        "MSL: {msl}"
    );
    assert!(
        msl.contains("bool p0 = false, p1 = false, p2 = false, p3 = false;"),
        "MSL: {msl}"
    );
    assert!(
        msl.contains("threadgroup uint8_t local_mem[1024];"),
        "MSL: {msl}"
    );
}

#[test]
fn test_device_atomic_add() {
    // DeviceAtomic opcode 0x3D, scope in word1 bits 0-1
    let code = extended_scope(0x3D, 5, 3, 4, 0, 2);
    let msl = compile_instructions(&code);
    assert!(msl.contains("atomic_fetch_add_explicit"), "MSL: {msl}");
    assert!(
        msl.contains("(device atomic_uint*)(device_mem + r3)"),
        "MSL: {msl}"
    );
    assert!(msl.contains("memory_order_relaxed"), "MSL: {msl}");
}

#[test]
fn test_fence() {
    // FenceAcquire = SyncOp::FenceAcquire(3) + SYNC_MODIFIER_OFFSET(8) = 11
    // Fences read word1 for scope; scope=1 (Workgroup) in word1 bits 0-1
    let mut code = encode_word0(0x3F, 0, 0, 3 + SYNC_MODIFIER_OFFSET, 0, false)
        .to_le_bytes()
        .to_vec();
    code.extend_from_slice(&(1u32 << EXTENDED_SCOPE_SHIFT).to_le_bytes());
    let msl = compile_instructions(&code);
    assert!(
        msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"),
        "MSL: {msl}"
    );
}

#[test]
fn test_nop_produces_no_output() {
    // Nop = SyncOp::Nop(7) + SYNC_MODIFIER_OFFSET(8) = 15
    // Nop does NOT read word1
    let code = single(0x3F, 0, 0, 7 + SYNC_MODIFIER_OFFSET);
    let msl = compile_instructions(&code);
    let lines: Vec<&str> = msl.lines().collect();
    let return_count = lines.iter().filter(|l| l.trim() == "return;").count();
    assert_eq!(return_count, 1, "should only have one return from halt");
}

#[test]
fn test_f16_hadd() {
    let code = extended(0x1C, 3, 1, 2, 0);
    let msl = compile_instructions(&code);
    assert!(msl.contains("rhi(rh(r1) + rh(r2))"), "MSL: {msl}");
}
