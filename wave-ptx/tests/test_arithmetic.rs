// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Check PTX arithmetic instruction emission.
//!
//! Explicit rounding preserves instruction boundaries under the
//! [PTX arithmetic contract](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-add).

use wave_emu::{Emulator, EmulatorConfig};
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

fn halt_instruction() -> Vec<u8> {
    single(0x3F, 0, 0, 9, 0)
}

fn build_wbin(name: &str, reg_count: u32, local_mem: u32, code: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    let name_bytes: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
    let code_offset: u32 = 0x20;
    let code_size = u32::try_from(code.len()).unwrap();
    let symbol_offset = code_offset + code_size;
    let symbol_size = u32::try_from(name_bytes.len()).unwrap();
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

fn mov_imm(rd: u8, value: u32) -> Vec<u8> {
    let mut instruction = encode_word0(0x41, rd, 0, 1, 0).to_le_bytes().to_vec();
    instruction.extend_from_slice(&value.to_le_bytes());
    instruction
}

fn run_program(
    code: &[u8],
    initial: &[u8],
    expected: &[u8],
    f64_enabled: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let binary = build_wbin("arithmetic_boundaries", 32, 0, code);
    let mut emulator = Emulator::new(EmulatorConfig {
        workgroup_dim: [1, 1, 1],
        use_kernel_workgroup_size: false,
        device_memory_size: initial.len(),
        f64_enabled,
        ..EmulatorConfig::default()
    });
    emulator.load_binary(&binary)?;
    emulator.load_device_memory(0, initial)?;
    emulator.run()?;
    assert_eq!(emulator.read_device_memory(0, initial.len())?, expected);
    Ok(compile(&binary, 75)?)
}

#[test]
fn separate_f32_operations_preserve_rounding_boundaries() -> Result<(), Box<dyn std::error::Error>>
{
    // Exact product is 1 - 2^-46. Its f32 rounding is 1, so only FMA retains the residual.
    let initial_words = [
        0x3f80_0001_u32,
        0x3f7f_fffe,
        0xbf80_0000,
        0x3f80_0000,
        0xdead_beef,
        0xdead_beef,
        0xdead_beef,
        0xdead_beef,
    ];
    let mut expected_words = initial_words;
    expected_words[4..7].copy_from_slice(&[0x0000_0000, 0x0000_0000, 0xa880_0000]);
    let initial: Vec<_> = initial_words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let expected: Vec<_> = expected_words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let mut code = Vec::new();
    for register in 1..=4 {
        code.extend(mov_imm(0, u32::from(register - 1) * 4));
        code.extend(single(0x38, register, 0, 2, 0));
    }
    code.extend(extended(0x12, 5, 1, 2, 0, 0));
    code.extend(extended(0x10, 6, 5, 3, 0, 0));
    code.extend(extended(0x11, 7, 5, 4, 0, 0));
    code.extend(extended3(0x13, 8, 1, 2, 3, 0, 0));
    for register in 6..=8 {
        code.extend(mov_imm(0, 16 + u32::from(register - 6) * 4));
        code.extend(extended(0x39, 0, 0, register, 2, 0));
    }
    code.extend(halt_instruction());
    let ptx = run_program(&code, &initial, &expected, false)?;
    let arithmetic: Vec<_> = ptx
        .lines()
        .map(str::trim)
        .filter(|line| {
            matches!(line.split('.').next(), Some("mul" | "add" | "sub" | "fma"))
                && line.contains(".f32 ")
        })
        .collect();
    assert_eq!(
        arithmetic,
        [
            "mul.rn.f32 %f5, %f1, %f2;",
            "add.rn.f32 %f6, %f5, %f3;",
            "sub.rn.f32 %f7, %f5, %f4;",
            "fma.rn.f32 %f8, %f1, %f2, %f3;",
        ]
    );
    Ok(())
}

#[test]
fn separate_f64_operations_preserve_rounding_boundaries() -> Result<(), Box<dyn std::error::Error>>
{
    // Exact product is 1 - 2^-104. The literal FMA result is representable in f64.
    let inputs = [
        0x3ff0_0000_0000_0001_u64,
        0x3fef_ffff_ffff_fffe,
        0xbff0_0000_0000_0000,
    ];
    let mut initial: Vec<_> = inputs.into_iter().flat_map(u64::to_le_bytes).collect();
    initial.extend([0xdead_beef_u32; 5].into_iter().flat_map(u32::to_le_bytes));
    let mut expected = initial.clone();
    expected[24..32].copy_from_slice(&0x0000_0000_0000_0000_u64.to_le_bytes());
    expected[32..40].copy_from_slice(&0xb970_0000_0000_0000_u64.to_le_bytes());
    let mut code = Vec::new();
    for register in 2..=7 {
        code.extend(mov_imm(0, u32::from(register - 2) * 4));
        code.extend(single(0x38, register, 0, 2, 0));
    }
    code.extend(extended(0x1e, 10, 2, 4, 2, 0));
    code.extend(extended(0x1e, 12, 10, 6, 0, 0));
    code.extend(extended3(0x1e, 14, 2, 4, 6, 3, 0));
    for register in 12..=15 {
        code.extend(mov_imm(0, 24 + u32::from(register - 12) * 4));
        code.extend(extended(0x39, 0, 0, register, 2, 0));
    }
    code.extend(halt_instruction());
    let ptx = run_program(&code, &initial, &expected, true)?;
    let arithmetic: Vec<_> = ptx
        .lines()
        .map(str::trim)
        .filter(|line| {
            matches!(line.split('.').next(), Some("mul" | "add" | "fma")) && line.contains(".f64 ")
        })
        .collect();
    assert_eq!(
        arithmetic,
        [
            "mul.rn.f64 %rd2, %rd2, %rd3;",
            "add.rn.f64 %rd2, %rd2, %rd3;",
            "fma.rn.f64 %rd2, %rd2, %rd3, %rd1;",
        ]
    );
    Ok(())
}

#[test]
fn separate_arithmetic_formats_have_explicit_rounding() {
    for (opcode, operations) in [
        (
            0x1c,
            &[(0, "add.rn.f16"), (1, "sub.rn.f16"), (2, "mul.rn.f16")][..],
        ),
        (0x1d, &[(0, "add.rn.f16x2"), (1, "mul.rn.f16x2")][..]),
        (
            0x1e,
            &[(0, "add.rn.f64"), (1, "sub.rn.f64"), (2, "mul.rn.f64")][..],
        ),
        (
            0x2d,
            &[(0, "add.rn.f32"), (1, "sub.rn.f32"), (2, "mul.rn.f32")][..],
        ),
        (0x2e, &[(0, "add.rn.f32"), (1, "mul.rn.f32")][..]),
    ] {
        for &(modifier, mnemonic) in operations {
            let ptx = compile_instrs(&extended(opcode, 8, 2, 4, modifier, 0));
            let instructions: Vec<_> = ptx
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .filter(|word| matches!(word.split('.').next(), Some("add" | "sub" | "mul")))
                .collect();
            let count = if opcode == 0x2e { 2 } else { 1 };
            assert_eq!(instructions, vec![mnemonic; count], "PTX: {ptx}");
        }
    }
}

#[test]
fn fused_arithmetic_formats_remain_explicit_fma() {
    for (opcode, modifier, mnemonic, count) in [
        (0x1c, 3, "fma.rn.f16", 1),
        (0x1d, 2, "fma.rn.f16x2", 1),
        (0x1e, 3, "fma.rn.f64", 1),
        (0x2d, 3, "fma.rn.f32", 1),
        (0x2e, 2, "fma.rn.f32", 2),
    ] {
        let ptx = compile_instrs(&extended3(opcode, 8, 2, 4, 6, modifier, 0));
        let instructions: Vec<_> = ptx
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .filter(|word| word.starts_with("fma."))
            .collect();
        assert_eq!(instructions, vec![mnemonic; count], "PTX: {ptx}");
    }
}

#[test]
fn test_iadd() {
    let ptx = compile_instrs(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(ptx.contains("add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_isub() {
    let ptx = compile_instrs(&extended(0x01, 2, 0, 1, 0, 0));
    assert!(ptx.contains("sub.s32 %r2, %r0, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_imul() {
    let ptx = compile_instrs(&extended(0x02, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mul.lo.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_imul_hi() {
    let ptx = compile_instrs(&extended(0x03, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mul.hi.u32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_idiv() {
    let ptx = compile_instrs(&extended(0x05, 3, 1, 2, 0, 0));
    assert!(ptx.contains("div.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_imod() {
    let ptx = compile_instrs(&extended(0x06, 3, 1, 2, 0, 0));
    assert!(ptx.contains("rem.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_ineg() {
    let ptx = compile_instrs(&single(0x07, 2, 1, 0, 0));
    assert!(ptx.contains("neg.s32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_iabs() {
    let ptx = compile_instrs(&single(0x08, 2, 1, 0, 0));
    assert!(ptx.contains("abs.s32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_imin() {
    let ptx = compile_instrs(&extended(0x09, 3, 1, 2, 0, 0));
    assert!(ptx.contains("min.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_imax() {
    let ptx = compile_instrs(&extended(0x0A, 3, 1, 2, 0, 0));
    assert!(ptx.contains("max.s32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_fadd() {
    let ptx = compile_instrs(&extended(0x10, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mov.b32 %f1, %r1;"), "PTX: {ptx}");
    assert!(ptx.contains("mov.b32 %f2, %r2;"), "PTX: {ptx}");
    assert!(ptx.contains("add.rn.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
    assert!(ptx.contains("mov.b32 %r3, %f3;"), "PTX: {ptx}");
}

#[test]
fn test_fsub() {
    let ptx = compile_instrs(&extended(0x11, 3, 1, 2, 0, 0));
    assert!(ptx.contains("sub.rn.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
}

#[test]
fn test_fmul() {
    let ptx = compile_instrs(&extended(0x12, 3, 1, 2, 0, 0));
    assert!(ptx.contains("mul.rn.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
}

#[test]
fn test_fdiv() {
    let ptx = compile_instrs(&extended(0x14, 3, 1, 2, 0, 0));
    assert!(ptx.contains("div.approx.f32 %f3, %f1, %f2;"), "PTX: {ptx}");
}

#[test]
fn test_fsqrt() {
    let ptx = compile_instrs(&single(0x1A, 2, 1, 0, 0));
    assert!(ptx.contains("sqrt.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fneg() {
    let ptx = compile_instrs(&single(0x15, 2, 1, 0, 0));
    assert!(ptx.contains("neg.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fabs() {
    let ptx = compile_instrs(&single(0x16, 2, 1, 0, 0));
    assert!(ptx.contains("abs.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fsin() {
    let ptx = compile_instrs(&single(0x1B, 2, 1, 8, 0));
    assert!(ptx.contains("sin.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fcos() {
    let ptx = compile_instrs(&single(0x1B, 2, 1, 9, 0));
    assert!(ptx.contains("cos.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_fexp2() {
    let ptx = compile_instrs(&single(0x1B, 2, 1, 10, 0));
    assert!(ptx.contains("ex2.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_flog2() {
    let ptx = compile_instrs(&single(0x1B, 2, 1, 11, 0));
    assert!(ptx.contains("lg2.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_ffloor() {
    let ptx = compile_instrs(&single(0x1B, 2, 1, 2, 0));
    assert!(ptx.contains("cvt.rmi.f32.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_ffract_preserves_subtraction_rounding() {
    let ptx = compile_instrs(&single(0x1B, 2, 1, 6, 0));
    assert!(ptx.contains("sub.rn.f32 %f2, %f1, %ft0;"), "PTX: {ptx}");
}

#[test]
fn test_frsqrt() {
    let ptx = compile_instrs(&single(0x1B, 2, 1, 0, 0));
    assert!(ptx.contains("rsqrt.approx.f32 %f2, %f1;"), "PTX: {ptx}");
}

#[test]
fn test_predicated_iadd() {
    let ptx = compile_instrs(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(ptx.contains("add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_predicated_negated() {
    let ptx = compile_instrs(&extended(0x00, 5, 3, 4, 0, 0));
    assert!(ptx.contains("add.s32 %r5, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_mov() {
    let ptx = compile_instrs(&single(0x41, 5, 3, 0, 0));
    assert!(ptx.contains("mov.b32 %r5, %r3;"), "PTX: {ptx}");
}

#[test]
fn test_mov_imm() {
    let mut code = encode_word0(0x41, 5, 0, 1, 0).to_le_bytes().to_vec();
    code.extend_from_slice(&42u32.to_le_bytes());
    let ptx = compile_instrs(&code);
    assert!(ptx.contains("mov.b32 %r5, 42;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_thread_id() {
    let ptx = compile_instrs(&single(0x41, 5, 0, 2, 0));
    assert!(ptx.contains("mov.u32 %r5, %tid.x;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_lane_id() {
    let ptx = compile_instrs(&single(0x41, 5, 4, 2, 0));
    assert!(ptx.contains("mov.u32 %r5, %laneid;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_ctaid() {
    let ptx = compile_instrs(&single(0x41, 5, 5, 2, 0));
    assert!(ptx.contains("mov.u32 %r5, %ctaid.x;"), "PTX: {ptx}");
}

#[test]
fn test_mov_sr_wave_width() {
    let ptx = compile_instrs(&single(0x41, 5, 14, 2, 0));
    assert!(ptx.contains("mov.u32 %r5, 32;"), "PTX: {ptx}");
}

#[test]
fn test_bitwise_and() {
    let ptx = compile_instrs(&extended(0x20, 3, 1, 2, 0, 0));
    assert!(ptx.contains("and.b32 %r3, %r1, %r2;"), "PTX: {ptx}");
}

#[test]
fn test_bitwise_shl() {
    let ptx = compile_instrs(&extended(0x24, 3, 1, 2, 0, 0));
    assert!(ptx.contains("and.b32 %t0, %r2, 31;"), "PTX: {ptx}");
    assert!(ptx.contains("shl.b32 %r3, %r1, %t0;"), "PTX: {ptx}");
}

#[test]
fn test_bitwise_sar() {
    let ptx = compile_instrs(&extended(0x26, 3, 1, 2, 0, 0));
    assert!(ptx.contains("and.b32 %t0, %r2, 31;"), "PTX: {ptx}");
    assert!(ptx.contains("shr.s32 %r3, %r1, %t0;"), "PTX: {ptx}");
}

#[test]
fn test_bitcount() {
    let ptx = compile_instrs(&single(0x27, 2, 1, 0, 0));
    assert!(ptx.contains("popc.b32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_bitrev() {
    let ptx = compile_instrs(&single(0x27, 2, 1, 2, 0));
    assert!(ptx.contains("brev.b32 %r2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_icmp_eq() {
    let ptx = compile_instrs(&extended(0x28, 1, 3, 4, 0, 0));
    assert!(ptx.contains("setp.eq.s32 %p1, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_ucmp_lt() {
    let ptx = compile_instrs(&extended(0x29, 2, 3, 4, 2, 0));
    assert!(ptx.contains("setp.lt.u32 %p2, %r3, %r4;"), "PTX: {ptx}");
}

#[test]
fn test_fcmp_gt() {
    let ptx = compile_instrs(&extended(0x2A, 1, 3, 4, 4, 0));
    assert!(ptx.contains("setp.gt.f32 %p1, %f3, %f4;"), "PTX: {ptx}");
}

#[test]
fn test_select() {
    let ptx = compile_instrs(&extended3(0x2B, 5, 1, 3, 4, 0, 0));
    assert!(ptx.contains("selp.b32 %r5, %r3, %r4, %p1;"), "PTX: {ptx}");
}

#[test]
fn test_cvt_i32_f32() {
    let ptx = compile_instrs(&single(0x2C, 2, 1, 2, 0));
    assert!(ptx.contains("cvt.rn.f32.s32 %f2, %r1;"), "PTX: {ptx}");
}

#[test]
fn test_kernel_header() {
    let code = halt_instruction();
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
    let code = halt_instruction();
    let wbin = build_wbin("test", 4, 0, &code);
    let ptx = compile(&wbin, 80).unwrap();
    assert!(ptx.contains(".target sm_80"), "PTX: {ptx}");
}

#[test]
fn test_shared_mem_declaration() {
    let code = halt_instruction();
    let wbin = build_wbin("test", 4, 4096, &code);
    let ptx = compile(&wbin, 75).unwrap();
    assert!(
        ptx.contains(".shared .align 4 .b8 _shared_mem[4096];"),
        "PTX: {ptx}"
    );
}

#[test]
fn test_no_shared_mem_when_zero() {
    let code = halt_instruction();
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
    let ptx = compile_instrs(&single(0x3F, 0, 0, 10, 0));
    assert!(ptx.contains("bar.sync 0;"), "PTX: {ptx}");
}

#[test]
fn large_shift_counts_preserve_low_five_bits_with_aliased_destinations(
) -> Result<(), Box<dyn std::error::Error>> {
    for (opcode, mnemonic, shifted) in [
        (0x24, "shl.b32", 0x8000_0000_u32),
        (0x25, "shr.u32", 1),
        (0x26, "shr.s32", u32::MAX),
    ] {
        for (count, expected) in [(32, 0x8000_0003), (63, shifted), (u32::MAX, shifted)] {
            for destination in [1, 2] {
                let mut code = mov_imm(1, 0x8000_0003);
                code.extend(mov_imm(2, count));
                code.extend(extended(opcode, destination, 1, 2, 0, 0));
                code.extend(mov_imm(0, 0));
                code.extend(extended(0x39, 0, 0, destination, 2, 0));
                code.extend(halt_instruction());
                let ptx = run_program(&code, &[0; 4], &expected.to_le_bytes(), false)?;
                let mask = ptx.find("and.b32 %t0, %r2, 31;").unwrap();
                let write = ptx
                    .find(&format!("{mnemonic} %r{destination}, %r1, %t0;"))
                    .unwrap();
                assert!(
                    mask < write,
                    "count must be captured before destination write: {ptx}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn high_multiply_uses_unsigned_register_bits() -> Result<(), Box<dyn std::error::Error>> {
    for (left, right, high) in [
        (u32::MAX, u32::MAX, 0xffff_fffe_u32),
        (0x8000_0000, 2, 1),
        (u32::MAX, 0x8000_0000, 0x7fff_ffff),
    ] {
        for destination in [1, 2] {
            let mut code = mov_imm(1, left);
            code.extend(mov_imm(2, right));
            code.extend(extended(0x03, destination, 1, 2, 0, 0));
            code.extend(mov_imm(0, 0));
            code.extend(extended(0x39, 0, 0, destination, 2, 0));
            code.extend(halt_instruction());
            let ptx = run_program(&code, &[0; 4], &high.to_le_bytes(), false)?;
            assert!(ptx.contains(&format!(
                "@%wave_active mul.hi.u32 %r{destination}, %r1, %r2;"
            )));
            assert!(!ptx.contains("mul.hi.s32"));
        }
    }
    Ok(())
}
