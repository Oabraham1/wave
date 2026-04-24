// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for individual instruction codegen. Constructs minimal WBIN files with
//!
//! specific WAVE instructions, compiles them to HIP, and verifies the output contains
//! expected HIP code fragments. Tests cover integer arithmetic, float operations,
//! bitwise ops, memory access, control flow, special registers, and wave intrinsics.

use wave_hip::compile;

const OPCODE_SHIFT: u32 = 24;
const RD_SHIFT: u32 = 16;
const RS1_SHIFT: u32 = 8;
const MODIFIER_SHIFT: u32 = 4;
const EXTENDED_RS2_SHIFT: u32 = 24;
const EXTENDED_RS3_SHIFT: u32 = 16;
const EXTENDED_SCOPE_SHIFT: u32 = 0;
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

fn extended3(opcode: u8, rd: u8, rs1: u8, rs2: u8, rs3: u8, modifier: u8, pred: u8) -> Vec<u8> {
    let mut v = encode_word0(opcode, rd, rs1, modifier, pred)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(
        &((u32::from(rs2) << EXTENDED_RS2_SHIFT) | (u32::from(rs3) << EXTENDED_RS3_SHIFT))
            .to_le_bytes(),
    );
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
    v.extend_from_slice(
        &((u32::from(rs2) << EXTENDED_RS2_SHIFT) | (u32::from(scope) << EXTENDED_SCOPE_SHIFT))
            .to_le_bytes(),
    );
    v
}

fn sync_single(modifier_offset: u8) -> Vec<u8> {
    single(0x3F, 0, 0, modifier_offset + SYNC_MODIFIER_OFFSET, 0)
}

fn sync_extended_scope(modifier_offset: u8, scope: u8) -> Vec<u8> {
    let mut v = single(0x3F, 0, 0, modifier_offset + SYNC_MODIFIER_OFFSET, 0);
    v.extend_from_slice(&(u32::from(scope) << EXTENDED_SCOPE_SHIFT).to_le_bytes());
    v
}

fn halt_instruction() -> Vec<u8> {
    // Halt = SyncOp::Halt(1) + SYNC_MODIFIER_OFFSET(8) = modifier 9
    sync_single(1)
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
    let hip = compile_instructions(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(hip.contains("r5 = r3 + r4;"), "HIP: {hip}");
}

#[test]
fn test_isub() {
    let hip = compile_instructions(&extended(0x01, 2, 0, 1, 0, 0));
    assert!(hip.contains("r2 = r0 - r1;"), "HIP: {hip}");
}

#[test]
fn test_imul() {
    let hip = compile_instructions(&extended(0x02, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = r1 * r2;"), "HIP: {hip}");
}

#[test]
fn test_idiv() {
    let hip = compile_instructions(&extended(0x05, 3, 1, 2, 0, 0));
    assert!(hip.contains("(int32_t)r1 / (int32_t)r2"), "HIP: {hip}");
}

#[test]
fn test_ineg() {
    let hip = compile_instructions(&single(0x07, 2, 1, 0, 0));
    assert!(hip.contains("(uint32_t)(-(int32_t)r1)"), "HIP: {hip}");
}

#[test]
fn test_fadd() {
    let hip = compile_instructions(&extended(0x10, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = ri(rf(r1) + rf(r2));"), "HIP: {hip}");
}

#[test]
fn test_fsub() {
    let hip = compile_instructions(&extended(0x11, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = ri(rf(r1) - rf(r2));"), "HIP: {hip}");
}

#[test]
fn test_fmul() {
    let hip = compile_instructions(&extended(0x12, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = ri(rf(r1) * rf(r2));"), "HIP: {hip}");
}

#[test]
fn test_fsqrt() {
    let hip = compile_instructions(&single(0x1A, 2, 1, 0, 0));
    assert!(hip.contains("r2 = ri(sqrtf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_fsin() {
    let hip = compile_instructions(&single(0x1B, 2, 1, 8, 0));
    assert!(hip.contains("r2 = ri(__sinf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_fcos() {
    let hip = compile_instructions(&single(0x1B, 2, 1, 9, 0));
    assert!(hip.contains("r2 = ri(__cosf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_fexp2() {
    let hip = compile_instructions(&single(0x1B, 2, 1, 10, 0));
    assert!(hip.contains("r2 = ri(exp2f(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_flog2() {
    let hip = compile_instructions(&single(0x1B, 2, 1, 11, 0));
    assert!(hip.contains("r2 = ri(log2f(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_ffloor() {
    let hip = compile_instructions(&single(0x1B, 2, 1, 2, 0));
    assert!(hip.contains("r2 = ri(floorf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_frsqrt() {
    let hip = compile_instructions(&single(0x1B, 2, 1, 0, 0));
    assert!(hip.contains("r2 = ri(rsqrtf(rf(r1)));"), "HIP: {hip}");
}

#[test]
fn test_and() {
    let hip = compile_instructions(&extended(0x20, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = r1 & r2;"), "HIP: {hip}");
}

#[test]
fn test_shl() {
    let hip = compile_instructions(&extended(0x24, 3, 1, 2, 0, 0));
    assert!(hip.contains("r3 = r1 << (r2 & 0x1Fu);"), "HIP: {hip}");
}

#[test]
fn test_bitcount() {
    let hip = compile_instructions(&single(0x27, 2, 1, 0, 0));
    assert!(hip.contains("r2 = __popc(r1);"), "HIP: {hip}");
}

#[test]
fn test_bitrev() {
    let hip = compile_instructions(&single(0x27, 2, 1, 2, 0));
    assert!(hip.contains("r2 = __brev(r1);"), "HIP: {hip}");
}

#[test]
fn test_bitfind() {
    let hip = compile_instructions(&single(0x27, 2, 1, 1, 0));
    assert!(hip.contains("__ffs"), "HIP: {hip}");
}

#[test]
fn test_mov() {
    // MiscOp uses opcode 0x41, modifier 0 = Mov (single-word, decoder doesn't read word1)
    let hip = compile_instructions(&single(0x41, 5, 3, 0, 0));
    assert!(hip.contains("r5 = r3;"), "HIP: {hip}");
}

#[test]
fn test_mov_imm() {
    // MiscOp uses opcode 0x41, modifier 1 = MovImm; decoder reads word1 as immediate
    let word0 = encode_word0(0x41, 5, 0, 1, 0);
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&42u32.to_le_bytes());
    let hip = compile_instructions(&code);
    assert!(hip.contains("r5 = 42u;"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_thread_id() {
    // MiscOp uses opcode 0x41, modifier 2 = MovSr (single-word)
    let hip = compile_instructions(&single(0x41, 5, 0, 2, 0));
    assert!(hip.contains("r5 = (uint32_t)threadIdx.x;"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_lane_id() {
    let hip = compile_instructions(&single(0x41, 5, 4, 2, 0));
    assert!(hip.contains("r5 = (uint32_t)__lane_id();"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_block_idx() {
    let hip = compile_instructions(&single(0x41, 5, 5, 2, 0));
    assert!(hip.contains("r5 = (uint32_t)blockIdx.x;"), "HIP: {hip}");
}

#[test]
fn test_mov_sr_warp_size() {
    let hip = compile_instructions(&single(0x41, 5, 14, 2, 0));
    assert!(hip.contains("r5 = (uint32_t)warpSize;"), "HIP: {hip}");
}

#[test]
fn test_device_load_u32() {
    let hip = compile_instructions(&single(0x38, 5, 3, 2, 0));
    assert!(
        hip.contains("r5 = (uint32_t)(*(uint32_t*)(device_mem + r3));"),
        "HIP: {hip}"
    );
}

#[test]
fn test_device_store_u32() {
    let hip = compile_instructions(&extended(0x39, 0, 3, 5, 2, 0));
    assert!(
        hip.contains("*(uint32_t*)(device_mem + r3) = (uint32_t)r5;"),
        "HIP: {hip}"
    );
}

#[test]
fn test_shared_load_u32() {
    let hip = compile_instructions_with_shared(&single(0x30, 5, 3, 2, 0), 4096);
    assert!(
        hip.contains("r5 = (uint32_t)(*(uint32_t*)(local_mem + r3));"),
        "HIP: {hip}"
    );
    assert!(
        hip.contains("extern __shared__ uint8_t local_mem[];"),
        "HIP: {hip}"
    );
}

#[test]
fn test_barrier() {
    // Barrier = SyncOp::Barrier(2) + offset 8 = modifier 10
    let hip = compile_instructions(&sync_single(2));
    assert!(hip.contains("__syncthreads();"), "HIP: {hip}");
}

#[test]
fn test_fence_block() {
    // FenceAcquire = SyncOp::FenceAcquire(3) + offset 8 = modifier 11
    // Scope goes in word1 bits 0-1: Workgroup = 1
    let hip = compile_instructions(&sync_extended_scope(3, 1));
    assert!(hip.contains("__threadfence_block();"), "HIP: {hip}");
}

#[test]
fn test_fence_device() {
    // FenceAcquire = SyncOp::FenceAcquire(3) + offset 8 = modifier 11
    // Scope goes in word1 bits 0-1: Device = 2
    let hip = compile_instructions(&sync_extended_scope(3, 2));
    assert!(hip.contains("__threadfence();"), "HIP: {hip}");
}

#[test]
fn test_icmp_eq() {
    let hip = compile_instructions(&extended(0x28, 1, 3, 4, 0, 0));
    assert!(
        hip.contains("p1 = (int32_t)r3 == (int32_t)r4;"),
        "HIP: {hip}"
    );
}

#[test]
fn test_select() {
    let hip = compile_instructions(&extended3(0x2B, 5, 1, 3, 4, 0, 0));
    assert!(hip.contains("r5 = p1 ? r3 : r4;"), "HIP: {hip}");
}

#[test]
fn test_predicated_instruction() {
    let hip = compile_instructions(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(hip.contains("r5 = r3 + r4;"), "HIP: {hip}");
}

#[test]
fn test_predicated_negated() {
    let hip = compile_instructions(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(hip.contains("r5 = r3 + r4;"), "HIP: {hip}");
}

#[test]
fn test_if_else_endif() {
    let mut code = Vec::new();
    // Control flow ops use opcode 0x3F with modifier < 8
    // If=0, Else=1, Endif=2
    code.extend_from_slice(&single(0x3F, 0, 1, 0, 0)); // if p1
    code.extend_from_slice(&extended(0x00, 5, 3, 4, 0, 0)); // iadd
    code.extend_from_slice(&single(0x3F, 0, 0, 1, 0)); // else
    code.extend_from_slice(&extended(0x00, 5, 1, 2, 0, 0)); // iadd
    code.extend_from_slice(&single(0x3F, 0, 0, 2, 0)); // endif
    let hip = compile_instructions(&code);
    assert!(hip.contains("if (p1) {"), "HIP: {hip}");
    assert!(hip.contains("} else {"), "HIP: {hip}");
}

#[test]
fn test_loop_break() {
    let mut code = Vec::new();
    // Loop=3, Break=4, Endloop=6
    code.extend_from_slice(&single(0x3F, 0, 0, 3, 0)); // loop
    code.extend_from_slice(&single(0x3F, 0, 1, 4, 0)); // break p1
    code.extend_from_slice(&single(0x3F, 0, 0, 6, 0)); // endloop
    let hip = compile_instructions(&code);
    assert!(hip.contains("while (true) {"), "HIP: {hip}");
    assert!(hip.contains("if (p1) break;"), "HIP: {hip}");
}

#[test]
fn test_wave_shuffle() {
    let hip = compile_instructions(&extended(0x3E, 3, 1, 2, 0, 0));
    assert!(hip.contains("__shfl((int)r1, r2)"), "HIP: {hip}");
}

#[test]
fn test_wave_shuffle_xor() {
    let hip = compile_instructions(&extended(0x3E, 3, 1, 2, 3, 0));
    assert!(hip.contains("__shfl_xor((int)r1, r2)"), "HIP: {hip}");
}

#[test]
fn test_wave_ballot() {
    let hip = compile_instructions(&single(0x3E, 3, 1, 5, 0));
    assert!(hip.contains("(uint32_t)__ballot((int)p1)"), "HIP: {hip}");
}

#[test]
fn test_wave_any() {
    let hip = compile_instructions(&single(0x3E, 2, 1, 6, 0));
    assert!(hip.contains("p2 = __any((int)p1);"), "HIP: {hip}");
}

#[test]
fn test_wave_reduce_add() {
    let hip = compile_instructions(&single(0x3E, 3, 1, 9, 0));
    assert!(hip.contains("__shfl_down"), "HIP: {hip}");
    assert!(hip.contains("warpSize"), "HIP: {hip}");
}

#[test]
fn test_wave_prefix_sum() {
    let hip = compile_instructions(&single(0x3E, 3, 1, 8, 0));
    assert!(hip.contains("__shfl_up"), "HIP: {hip}");
    assert!(hip.contains("__lane_id()"), "HIP: {hip}");
}

#[test]
fn test_device_atomic_add() {
    let hip = compile_instructions(&extended_scope(0x3D, 5, 3, 4, 0, 2, 0));
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
    assert!(
        hip.contains("__device__ inline float rf(uint32_t r)"),
        "HIP: {hip}"
    );
    assert!(hip.contains("__uint_as_float"), "HIP: {hip}");
    assert!(hip.contains("__float_as_uint"), "HIP: {hip}");
}

#[test]
fn test_register_declarations() {
    let code = halt_instruction();
    let wbin = build_wbin("test_kernel", 4, 1024, &code);
    let hip = compile(&wbin).unwrap();
    assert!(
        hip.contains("uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;"),
        "HIP: {hip}"
    );
    assert!(
        hip.contains("bool p0 = false, p1 = false, p2 = false, p3 = false;"),
        "HIP: {hip}"
    );
    assert!(
        hip.contains("extern __shared__ uint8_t local_mem[];"),
        "HIP: {hip}"
    );
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
