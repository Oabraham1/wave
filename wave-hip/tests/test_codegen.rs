// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for individual instruction codegen. Constructs minimal WBIN files with
//!
//! specific WAVE instructions, compiles them to HIP, and verifies the output contains
//! expected HIP code fragments. Tests cover integer arithmetic, float operations,
//! bitwise ops, memory access, control flow, special registers, and wave intrinsics.

use wave_hip::compile;

const OPCODE_SHIFT: u32 = 26;
const RD_SHIFT: u32 = 21;
const RS1_SHIFT: u32 = 16;
const RS2_SHIFT: u32 = 11;
const MODIFIER_SHIFT: u32 = 7;
const SCOPE_SHIFT: u32 = 5;
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

fn encode_with_scope(
    opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, scope: u8, flags: u8,
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

fn encode_predicated(
    opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, flags: u8, pred: u8, negated: bool,
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

fn halt_instruction() -> [u8; 4] {
    encode(0x3F, 0, 0, 0, 1, SYNC_OP_FLAG)
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

fn compile_instructions_with_shared(instructions: &[u8], local_mem: u32) -> String {
    let mut code = instructions.to_vec();
    code.extend_from_slice(&halt_instruction());
    let wbin = build_wbin("test_kernel", 16, local_mem, &code);
    compile(&wbin).expect("compilation failed")
}

#[test]
fn test_iadd() {
    let hip = compile_instructions(&encode(0x00, 5, 3, 4, 0, 0));
    assert!(hip.contains("r5 = r3 + r4;"), "HIP: {hip}");
}

#[test]
fn test_isub() {
    let hip = compile_instructions(&encode(0x01, 2, 0, 1, 0, 0));
    assert!(hip.contains("r2 = r0 - r1;"), "HIP: {hip}");
}

#[test]
fn test_imul() {
    let hip = compile_instructions(&encode(0x02, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = r1 * r2;"), "HIP: {hip}");
}

#[test]
fn test_idiv() {
    let hip = compile_instructions(&encode(0x05, 3, 1, 2, 0, 0));
    assert!(hip.contains("(int32_t)r1 / (int32_t)r2"), "HIP: {hip}");
}

#[test]
fn test_ineg() {
    let hip = compile_instructions(&encode(0x07, 2, 1, 0, 0, 0));
    assert!(hip.contains("(uint32_t)(-(int32_t)r1)"), "HIP: {hip}");
}

#[test]
fn test_fadd() {
    let hip = compile_instructions(&encode(0x10, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = ri(rf(r1) + rf(r2));"), "HIP: {hip}");
}

#[test]
fn test_fsub() {
    let hip = compile_instructions(&encode(0x11, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = ri(rf(r1) - rf(r2));"), "HIP: {hip}");
}

#[test]
fn test_fmul() {
    let hip = compile_instructions(&encode(0x12, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = ri(rf(r1) * rf(r2));"), "HIP: {hip}");
}

#[test]
fn test_fsqrt() {
    let hip = compile_instructions(&encode(0x1A, 2, 1, 0, 0, 0));
    assert!(hip.contains("r2 = ri(sqrtf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_fsin() {
    let hip = compile_instructions(&encode(0x1B, 2, 1, 0, 8, 0));
    assert!(hip.contains("r2 = ri(__sinf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_fcos() {
    let hip = compile_instructions(&encode(0x1B, 2, 1, 0, 9, 0));
    assert!(hip.contains("r2 = ri(__cosf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_fexp2() {
    let hip = compile_instructions(&encode(0x1B, 2, 1, 0, 10, 0));
    assert!(hip.contains("r2 = ri(exp2f(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_flog2() {
    let hip = compile_instructions(&encode(0x1B, 2, 1, 0, 11, 0));
    assert!(hip.contains("r2 = ri(log2f(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_ffloor() {
    let hip = compile_instructions(&encode(0x1B, 2, 1, 0, 2, 0));
    assert!(hip.contains("r2 = ri(floorf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_frsqrt() {
    let hip = compile_instructions(&encode(0x1B, 2, 1, 0, 0, 0));
    assert!(hip.contains("r2 = ri(rsqrtf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_and() {
    let hip = compile_instructions(&encode(0x20, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = r1 & r2;"), "HIP: {hip}");
}

#[test]
fn test_shl() {
    let hip = compile_instructions(&encode(0x24, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = r1 << (r2 & 0x1Fu);"), "HIP: {hip}");
}

#[test]
fn test_bitcount() {
    let mut code = encode(0x27, 2, 1, 0, 0, 0).to_vec();
    code.extend_from_slice(&[0u8; 4]);
    let hip = compile_instructions(&code);
    assert!(hip.contains("r2 = __popc(r1);"), "HIP: {hip}");
}

#[test]
fn test_bitrev() {
    let mut code = encode(0x27, 2, 1, 0, 2, 0).to_vec();
    code.extend_from_slice(&[0u8; 4]);
    let hip = compile_instructions(&code);
    assert!(hip.contains("r2 = __brev(r1);"), "HIP: {hip}");
}

#[test]
fn test_bitfind() {
    let mut code = encode(0x27, 2, 1, 0, 1, 0).to_vec();
    code.extend_from_slice(&[0u8; 4]);
    let hip = compile_instructions(&code);
    assert!(hip.contains("__ffs"), "HIP: {hip}");
}

#[test]
fn test_mov() {
    let hip = compile_instructions(&encode(0x3F, 5, 3, 0, 0, MISC_OP_FLAG));
    assert!(hip.contains("r5 = r3;"), "HIP: {hip}");
}

#[test]
fn test_mov_imm() {
    let mut code = encode(0x3F, 5, 0, 0, 1, MISC_OP_FLAG).to_vec();
    code.extend_from_slice(&42u32.to_le_bytes());
    let hip = compile_instructions(&code);
    assert!(hip.contains("r5 = 42u;"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_thread_id() {
    let hip = compile_instructions(&encode(0x3F, 5, 0, 0, 2, MISC_OP_FLAG));
    assert!(hip.contains("r5 = (uint32_t)threadIdx.x;"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_lane_id() {
    let hip = compile_instructions(&encode(0x3F, 5, 4, 0, 2, MISC_OP_FLAG));
    assert!(hip.contains("r5 = (uint32_t)__lane_id();"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_block_idx() {
    let hip = compile_instructions(&encode(0x3F, 5, 5, 0, 2, MISC_OP_FLAG));
    assert!(hip.contains("r5 = (uint32_t)blockIdx.x;"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_warp_size() {
    let hip = compile_instructions(&encode(0x3F, 5, 14, 0, 2, MISC_OP_FLAG));
    assert!(hip.contains("r5 = (uint32_t)warpSize;"), "HIP: {hip}");
}

#[test]
fn test_device_load_u32() {
    let hip = compile_instructions(&encode(0x38, 5, 3, 0, 2, 0));
    assert!(
        hip.contains("r5 = (uint32_t)(*(uint32_t*)(device_mem + r3));"),
        "HIP: {hip}"
    );
}

#[test]
fn test_device_store_u32() {
    let hip = compile_instructions(&encode(0x39, 0, 3, 5, 2, 0));
    assert!(
        hip.contains("*(uint32_t*)(device_mem + r3) = (uint32_t)r5;"),
        "HIP: {hip}"
    );
}

#[test]
fn test_shared_load_u32() {
    let hip = compile_instructions_with_shared(&encode(0x30, 5, 3, 0, 2, 0), 4096);
    assert!(
        hip.contains("r5 = (uint32_t)(*(uint32_t*)(local_mem + r3));"),
        "HIP: {hip}"
    );
    assert!(hip.contains("extern __shared__ uint8_t local_mem[];"), "HIP: {hip}");
}

#[test]
fn test_barrier() {
    let hip = compile_instructions(&encode(0x3F, 0, 0, 0, 2, SYNC_OP_FLAG));
    assert!(hip.contains("__syncthreads();"), "HIP: {hip}");
}

#[test]
fn test_fence_block() {
    let hip = compile_instructions(&encode_with_scope(0x3F, 0, 0, 0, 3, 1, SYNC_OP_FLAG));
    assert!(hip.contains("__threadfence_block();"), "HIP: {hip}");
}

#[test]
fn test_fence_device() {
    let hip = compile_instructions(&encode_with_scope(0x3F, 0, 0, 0, 3, 2, SYNC_OP_FLAG));
    assert!(hip.contains("__threadfence();"), "HIP: {hip}");
}

#[test]
fn test_icmp_eq() {
    let hip = compile_instructions(&encode(0x28, 1, 3, 4, 0, 0));
    assert!(hip.contains("p1 = (int32_t)r3 == (int32_t)r4;"), "HIP: {hip}");
}

#[test]
fn test_select() {
    let hip = compile_instructions(&encode(0x2B, 5, 1, 3, 4, 0));
    assert!(hip.contains("r5 = p1 ? r3 : r4;"), "HIP: {hip}");
}

#[test]
fn test_predicated_instruction() {
    let hip = compile_instructions(&encode_predicated(0x00, 5, 3, 4, 0, 0, 1, false));
    assert!(hip.contains("if (p1) {"), "HIP: {hip}");
    assert!(hip.contains("r5 = r3 + r4;"), "HIP: {hip}");
}

#[test]
fn test_predicated_negated() {
    let hip = compile_instructions(&encode_predicated(0x00, 5, 3, 4, 0, 0, 2, true));
    assert!(hip.contains("if (!p2) {"), "HIP: {hip}");
}

#[test]
fn test_if_else_endif() {
    let mut code = Vec::new();
    code.extend_from_slice(&encode(0x3F, 0, 1, 0, 0, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x00, 5, 3, 4, 0, 0));
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 1, 0));
    code.extend_from_slice(&[0u8; 4]);
    code.extend_from_slice(&encode(0x00, 5, 1, 2, 0, 0));
    code.extend_from_slice(&encode(0x3F, 0, 0, 0, 2, 0));
    code.extend_from_slice(&[0u8; 4]);
    let hip = compile_instructions(&code);
    assert!(hip.contains("if (p1) {"), "HIP: {hip}");
    assert!(hip.contains("} else {"), "HIP: {hip}");
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
    let hip = compile_instructions(&code);
    assert!(hip.contains("while (true) {"), "HIP: {hip}");
    assert!(hip.contains("if (p1) break;"), "HIP: {hip}");
}

#[test]
fn test_wave_shuffle() {
    let hip = compile_instructions(&encode(0x3E, 3, 1, 2, 0, 0));
    assert!(hip.contains("__shfl((int)r1, r2)"), "HIP: {hip}");
}

#[test]
fn test_wave_shuffle_xor() {
    let hip = compile_instructions(&encode(0x3E, 3, 1, 2, 3, 0));
    assert!(hip.contains("__shfl_xor((int)r1, r2)"), "HIP: {hip}");
}

#[test]
fn test_wave_ballot() {
    let hip = compile_instructions(&encode(0x3E, 3, 1, 0, 5, 0));
    assert!(hip.contains("(uint32_t)__ballot((int)p1)"), "HIP: {hip}");
}

#[test]
fn test_wave_any() {
    let hip = compile_instructions(&encode(0x3E, 2, 1, 0, 6, 0));
    assert!(hip.contains("p2 = __any((int)p1);"), "HIP: {hip}");
}

#[test]
fn test_wave_reduce_add() {
    let hip = compile_instructions(&encode(0x3E, 3, 1, 0, 9, 0));
    assert!(hip.contains("__shfl_down"), "HIP: {hip}");
    assert!(hip.contains("warpSize"), "HIP: {hip}");
}

#[test]
fn test_wave_prefix_sum() {
    let hip = compile_instructions(&encode(0x3E, 3, 1, 0, 8, 0));
    assert!(hip.contains("__shfl_up"), "HIP: {hip}");
    assert!(hip.contains("__lane_id()"), "HIP: {hip}");
}

#[test]
fn test_device_atomic_add() {
    let mut code = encode_with_scope(0x3D, 5, 3, 4, 0, 2, 0).to_vec();
    code.extend_from_slice(&[0u8; 4]);
    let hip = compile_instructions(&code);
    assert!(hip.contains("atomicAdd"), "HIP: {hip}");
    assert!(hip.contains("device_mem"), "HIP: {hip}");
}

#[test]
fn test_kernel_header() {
    let code = halt_instruction();
    let wbin = build_wbin("my_kernel", 4, 0, &code);
    let hip = compile(&wbin).unwrap();
    assert!(hip.contains("#include <hip/hip_runtime.h>"), "HIP: {hip}");
    assert!(hip.contains("__global__ void my_kernel("), "HIP: {hip}");
    assert!(hip.contains("uint8_t* device_mem"), "HIP: {hip}");
    assert!(hip.contains("__device__ inline float rf(uint32_t r)"), "HIP: {hip}");
    assert!(hip.contains("__uint_as_float"), "HIP: {hip}");
    assert!(hip.contains("__float_as_uint"), "HIP: {hip}");
}

#[test]
fn test_register_declarations() {
    let code = halt_instruction();
    let wbin = build_wbin("test_kernel", 4, 1024, &code);
    let hip = compile(&wbin).unwrap();
    assert!(hip.contains("uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;"), "HIP: {hip}");
    assert!(hip.contains("bool p0 = false, p1 = false, p2 = false, p3 = false;"), "HIP: {hip}");
    assert!(hip.contains("extern __shared__ uint8_t local_mem[];"), "HIP: {hip}");
    assert!(hip.contains("wave_count"), "HIP: {hip}");
    assert!(hip.contains("warpSize"), "HIP: {hip}");
}

#[test]
fn test_no_shared_when_zero() {
    let code = halt_instruction();
    let wbin = build_wbin("test", 4, 0, &code);
    let hip = compile(&wbin).unwrap();
    assert!(!hip.contains("local_mem"), "HIP: {hip}");
}
