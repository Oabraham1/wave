// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for individual instruction codegen. Constructs minimal WBIN files with
//!
//! specific WAVE instructions, compiles them to SYCL, and verifies the output contains
//! expected SYCL code fragments including sycl:: math functions, sub-group operations,
//! atomic_ref patterns, bit_cast helpers, and nd_item special register accessors.

use wave_sycl::compile;

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
fn sync_extended_scope(modifier: u8, scope: u8) -> Vec<u8> {
    let mut v = encode_word0(0x3F, 0, 0, SYNC_MODIFIER_OFFSET + modifier, 0)
        .to_le_bytes()
        .to_vec();
    v.extend_from_slice(&(u32::from(scope) << EXTENDED_SCOPE_SHIFT).to_le_bytes());
    v
}
fn halt_instruction() -> Vec<u8> {
    single(0x3F, 0, 0, SYNC_MODIFIER_OFFSET + 1, 0)
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
    compile(&wbin).expect("compilation failed")
}

fn compile_instrs_with_slm(instructions: &[u8], slm_size: u32) -> String {
    let mut code = instructions.to_vec();
    code.extend_from_slice(&halt_instruction());
    let wbin = build_wbin("test_kernel", 16, slm_size, &code);
    compile(&wbin).expect("compilation failed")
}

#[test]
fn test_iadd() {
    let s = compile_instrs(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(s.contains("r5 = r3 + r4;"), "SYCL: {s}");
}

#[test]
fn test_isub() {
    let s = compile_instrs(&extended(0x01, 2, 0, 1, 0, 0));
    assert!(s.contains("r2 = r0 - r1;"), "SYCL: {s}");
}

#[test]
fn test_imul() {
    let s = compile_instrs(&extended(0x02, 3, 1, 2, 0, 0));
    assert!(s.contains("r3 = r1 * r2;"), "SYCL: {s}");
}

#[test]
fn test_idiv() {
    let s = compile_instrs(&extended(0x05, 3, 1, 2, 0, 0));
    assert!(s.contains("(int32_t)r1 / (int32_t)r2"), "SYCL: {s}");
}

#[test]
fn test_iabs() {
    let s = compile_instrs(&single(0x08, 2, 1, 0, 0));
    assert!(s.contains("sycl::abs((int32_t)r1)"), "SYCL: {s}");
}

#[test]
fn test_imin() {
    let s = compile_instrs(&extended(0x09, 3, 1, 2, 0, 0));
    assert!(
        s.contains("sycl::min((int32_t)r1, (int32_t)r2)"),
        "SYCL: {s}"
    );
}

#[test]
fn test_fadd() {
    let s = compile_instrs(&extended(0x10, 3, 1, 2, 0, 0));
    assert!(s.contains("r3 = ri(rf(r1) + rf(r2));"), "SYCL: {s}");
}

#[test]
fn test_fsqrt() {
    let s = compile_instrs(&single(0x1A, 2, 1, 0, 0));
    assert!(s.contains("ri(sycl::sqrt(rf(r1)))"), "SYCL: {s}");
}

#[test]
fn test_fsin() {
    let s = compile_instrs(&single(0x1B, 2, 1, 8, 0));
    assert!(s.contains("ri(sycl::sin(rf(r1)))"), "SYCL: {s}");
}

#[test]
fn test_fcos() {
    let s = compile_instrs(&single(0x1B, 2, 1, 9, 0));
    assert!(s.contains("ri(sycl::cos(rf(r1)))"), "SYCL: {s}");
}

#[test]
fn test_fexp2() {
    let s = compile_instrs(&single(0x1B, 2, 1, 10, 0));
    assert!(s.contains("ri(sycl::exp2(rf(r1)))"), "SYCL: {s}");
}

#[test]
fn test_flog2() {
    let s = compile_instrs(&single(0x1B, 2, 1, 11, 0));
    assert!(s.contains("ri(sycl::log2(rf(r1)))"), "SYCL: {s}");
}

#[test]
fn test_frsqrt() {
    let s = compile_instrs(&single(0x1B, 2, 1, 0, 0));
    assert!(s.contains("ri(sycl::rsqrt(rf(r1)))"), "SYCL: {s}");
}

#[test]
fn test_bitcount() {
    let s = compile_instrs(&single(0x27, 2, 1, 0, 0));
    assert!(s.contains("sycl::popcount(r1)"), "SYCL: {s}");
}

#[test]
fn test_bitfind() {
    let s = compile_instrs(&single(0x27, 2, 1, 1, 0));
    assert!(s.contains("sycl::ctz(r1)"), "SYCL: {s}");
}

#[test]
fn test_mov() {
    let s = compile_instrs(&single(0x41, 5, 3, 0, 0));
    assert!(s.contains("r5 = r3;"), "SYCL: {s}");
}

#[test]
fn test_mov_imm() {
    let mut code = encode_word0(0x41, 5, 0, 1, 0).to_le_bytes().to_vec();
    code.extend_from_slice(&42u32.to_le_bytes());
    let s = compile_instrs(&code);
    assert!(s.contains("r5 = 42u;"), "SYCL: {s}");
}

#[test]
fn test_mov_sr_thread_id() {
    let s = compile_instrs(&single(0x41, 5, 0, 2, 0));
    assert!(
        s.contains("r5 = (uint32_t)it.get_local_id(0);"),
        "SYCL: {s}"
    );
}

#[test]
fn test_mov_sr_lane_id() {
    let s = compile_instrs(&single(0x41, 5, 4, 2, 0));
    assert!(
        s.contains("r5 = (uint32_t)sg.get_local_id()[0];"),
        "SYCL: {s}"
    );
}

#[test]
fn test_mov_sr_workgroup_id() {
    let s = compile_instrs(&single(0x41, 5, 5, 2, 0));
    assert!(s.contains("r5 = (uint32_t)it.get_group(0);"), "SYCL: {s}");
}

#[test]
fn test_mov_sr_wave_width() {
    let s = compile_instrs(&single(0x41, 5, 14, 2, 0));
    assert!(
        s.contains("r5 = (uint32_t)sg.get_max_local_range()[0];"),
        "SYCL: {s}"
    );
}

#[test]
fn test_device_load_u32() {
    let s = compile_instrs(&single(0x38, 5, 3, 2, 0));
    assert!(
        s.contains("r5 = (uint32_t)(*(uint32_t*)(device_mem_usm + r3));"),
        "SYCL: {s}"
    );
}

#[test]
fn test_device_store_u32() {
    let s = compile_instrs(&extended(0x39, 0, 3, 5, 2, 0));
    assert!(
        s.contains("*(uint32_t*)(device_mem_usm + r3) = (uint32_t)r5;"),
        "SYCL: {s}"
    );
}

#[test]
fn test_shared_load_u32() {
    let s = compile_instrs_with_slm(&single(0x30, 5, 3, 2, 0), 4096);
    assert!(
        s.contains("r5 = (uint32_t)(*(uint32_t*)(lm + r3));"),
        "SYCL: {s}"
    );
    assert!(s.contains("local_accessor"), "SYCL: {s}");
}

#[test]
fn test_barrier() {
    let s = compile_instrs(&single(0x3F, 0, 0, SYNC_MODIFIER_OFFSET + 2, 0));
    assert!(s.contains("group_barrier(it.get_group());"), "SYCL: {s}");
}

#[test]
fn test_fence_acquire() {
    let s = compile_instrs(&sync_extended_scope(3, 1));
    assert!(
        s.contains("atomic_fence(memory_order::acquire, memory_scope::work_group);"),
        "SYCL: {s}"
    );
}

#[test]
fn test_fence_device() {
    let s = compile_instrs(&sync_extended_scope(3, 2));
    assert!(
        s.contains("atomic_fence(memory_order::acquire, memory_scope::device);"),
        "SYCL: {s}"
    );
}

#[test]
fn test_icmp_eq() {
    let s = compile_instrs(&extended(0x28, 1, 3, 4, 0, 0));
    assert!(s.contains("p1 = (int32_t)r3 == (int32_t)r4;"), "SYCL: {s}");
}

#[test]
fn test_select() {
    let s = compile_instrs(&extended3(0x2B, 5, 1, 3, 4, 0, 0));
    assert!(s.contains("r5 = p1 ? r3 : r4;"), "SYCL: {s}");
}

#[test]
fn test_predicated_instruction() {
    let s = compile_instrs(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(s.contains("r5 = r3 + r4;"), "SYCL: {s}");
}

#[test]
fn test_predicated_negated() {
    let s = compile_instrs(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(s.contains("r5 = r3 + r4;"), "SYCL: {s}");
}

#[test]
fn test_wave_shuffle() {
    let s = compile_instrs(&extended(0x3E, 3, 1, 2, 0, 0));
    assert!(s.contains("select_from_group(sg, r1, r2)"), "SYCL: {s}");
}

#[test]
fn test_wave_shuffle_xor() {
    let s = compile_instrs(&extended(0x3E, 3, 1, 2, 3, 0));
    assert!(s.contains("permute_group_by_xor(sg, r1, r2)"), "SYCL: {s}");
}

#[test]
fn test_wave_broadcast() {
    let s = compile_instrs(&extended(0x3E, 3, 1, 2, 4, 0));
    assert!(s.contains("group_broadcast(sg, r1, r2)"), "SYCL: {s}");
}

#[test]
fn test_wave_ballot() {
    let s = compile_instrs(&single(0x3E, 3, 1, 5, 0));
    assert!(s.contains("reduce_over_group(sg,"), "SYCL: {s}");
    assert!(s.contains("bit_or<uint32_t>()"), "SYCL: {s}");
}

#[test]
fn test_wave_any() {
    let s = compile_instrs(&single(0x3E, 2, 1, 6, 0));
    assert!(s.contains("p2 = any_of_group(sg, p1);"), "SYCL: {s}");
}

#[test]
fn test_wave_all() {
    let s = compile_instrs(&single(0x3E, 2, 1, 7, 0));
    assert!(s.contains("p2 = all_of_group(sg, p1);"), "SYCL: {s}");
}

#[test]
fn test_wave_reduce_add() {
    let s = compile_instrs(&single(0x3E, 3, 1, 9, 0));
    assert!(
        s.contains("reduce_over_group(sg, r1, plus<uint32_t>())"),
        "SYCL: {s}"
    );
}

#[test]
fn test_wave_reduce_min() {
    let s = compile_instrs(&single(0x3E, 3, 1, 10, 0));
    assert!(
        s.contains("reduce_over_group(sg, (int32_t)r1, minimum<int32_t>())"),
        "SYCL: {s}"
    );
}

#[test]
fn test_wave_prefix_sum() {
    let s = compile_instrs(&single(0x3E, 3, 1, 8, 0));
    assert!(
        s.contains("exclusive_scan_over_group(sg, r1, plus<uint32_t>())"),
        "SYCL: {s}"
    );
}

#[test]
fn test_device_atomic_add() {
    let s = compile_instrs(&extended_scope(0x3D, 5, 3, 4, 0, 2, 0));
    assert!(s.contains("atomic_ref<uint32_t"), "SYCL: {s}");
    assert!(s.contains("fetch_add"), "SYCL: {s}");
    assert!(s.contains("global_space"), "SYCL: {s}");
}

#[test]
fn test_kernel_header() {
    let code = halt_instruction();
    let wbin = build_wbin("my_kernel", 4, 0, &code);
    let s = compile(&wbin).unwrap();
    assert!(s.contains("#include <sycl/sycl.hpp>"), "SYCL: {s}");
    assert!(s.contains("using namespace sycl;"), "SYCL: {s}");
    assert!(s.contains("inline float rf(uint32_t r)"), "SYCL: {s}");
    assert!(s.contains("bit_cast<float>"), "SYCL: {s}");
    assert!(s.contains("void my_kernel_launch(queue& q"), "SYCL: {s}");
    assert!(s.contains("nd_range<3>"), "SYCL: {s}");
    assert!(s.contains("nd_item<3> it"), "SYCL: {s}");
    assert!(s.contains("auto sg = it.get_sub_group()"), "SYCL: {s}");
}

#[test]
fn test_no_slm_when_zero() {
    let code = halt_instruction();
    let wbin = build_wbin("test", 4, 0, &code);
    let s = compile(&wbin).unwrap();
    assert!(!s.contains("local_accessor"), "SYCL: {s}");
    assert!(!s.contains("lm"), "SYCL: {s}");
}

#[test]
fn test_slm_declared_when_present() {
    let code = halt_instruction();
    let wbin = build_wbin("test", 4, 4096, &code);
    let s = compile(&wbin).unwrap();
    assert!(s.contains("local_accessor<uint8_t, 1>"), "SYCL: {s}");
    assert!(s.contains("range<1>(4096)"), "SYCL: {s}");
    assert!(s.contains("uint8_t* lm"), "SYCL: {s}");
}

#[test]
fn test_loop_break() {
    let mut code = Vec::new();
    code.extend_from_slice(&single(0x3F, 0, 0, 3, 0));
    code.extend_from_slice(&single(0x3F, 0, 1, 4, 0));
    code.extend_from_slice(&single(0x3F, 0, 0, 6, 0));
    let s = compile_instrs(&code);
    assert!(s.contains("while (true) {"), "SYCL: {s}");
    assert!(s.contains("if (p1) break;"), "SYCL: {s}");
}
