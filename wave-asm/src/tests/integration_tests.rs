// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! End-to-end assembler tests. Each test feeds source text through the full
//!
//! pipeline (lexer, parser, encoder, output) and verifies binary format, error
//! handling, or warning generation.

use crate::{assemble, diagnostics::AssemblerError};

fn assemble_program(source: &str) -> Result<Vec<u8>, AssemblerError> {
    assemble(source, "test.wave").map(|r| r.binary)
}

#[test]
fn test_integration_simple_kernel() {
    let source = r#"
.kernel simple
.registers 16
    mov_imm r0, 42
    iadd r1, r0, r0
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());

    let binary = result.unwrap();
    assert!(!binary.is_empty());
    assert_eq!(&binary[0..4], b"WAVE");
}

#[test]
fn test_integration_with_labels() {
    let source = r#"
.kernel with_labels
.registers 8
    mov_imm r0, 0
    icmp_lt p0, r0, r1
    if p0
        iadd r0, r0, r1
    endif
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_memory_ops() {
    let source = r#"
.kernel memory_test
.registers 4
.local_memory 1024
    local_load_u32 r0, r1
    local_store_u32 r2, r3
    device_load_u32 r0, r1
    device_store_u32 r2, r3
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_predicated_instructions() {
    let source = r#"
.kernel predicated
.registers 4
    icmp_eq p0, r0, r1
    @p0 iadd r2, r0, r1
    @!p0 isub r2, r0, r1
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_wave_ops() {
    let source = r#"
.kernel wave_ops
.registers 4
    wave_shuffle r0, r1, r2
    wave_broadcast r0, r1, r2
    wave_reduce_add r0, r1
    icmp_eq p0, r0, r1
    wave_ballot r2, p0
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_floating_point() {
    let source = r#"
.kernel float_ops
.registers 8
    fadd r0, r1, r2
    fmul r3, r4, r5
    fma r6, r0, r1, r2
    fsqrt r7, r0
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_control_flow() {
    let source = r#"
.kernel control_flow
.registers 4
    mov_imm r0, 0
    mov_imm r1, 10
loop
    iadd r0, r0, r1
    icmp_ge p0, r0, r1
    break p0
endloop
    barrier
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_synchronization() {
    let source = r#"
.kernel sync_test
.registers 2
    barrier
    fence_acquire .workgroup
    fence_release .device
    wait
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_multiple_kernels() {
    let source = r#"
.kernel kernel1
.registers 4
    iadd r0, r1, r2
.end

.kernel kernel2
.registers 8
    fadd r0, r1, r2
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());

    let binary = result.unwrap();
    let metadata_offset =
        u32::from_le_bytes([binary[0x18], binary[0x19], binary[0x1a], binary[0x1b]]) as usize;
    let kernel_count = u32::from_le_bytes([
        binary[metadata_offset],
        binary[metadata_offset + 1],
        binary[metadata_offset + 2],
        binary[metadata_offset + 3],
    ]);
    assert_eq!(kernel_count, 2);
}

#[test]
fn test_integration_error_undefined_label() {
    let source = r#"
.kernel error_test
.registers 2
    call undefined_function
.end
"#;

    let result = assemble_program(source);
    assert!(matches!(result, Err(AssemblerError::UndefinedLabel { .. })));
}

#[test]
fn test_integration_error_unclosed_kernel() {
    let source = r#"
.kernel unclosed
.registers 2
    nop
"#;

    let result = assemble_program(source);
    assert!(matches!(result, Err(AssemblerError::UnclosedKernel { .. })));
}

#[test]
fn test_integration_error_instruction_outside_kernel() {
    let source = r#"
iadd r0, r1, r2
"#;

    let result = assemble_program(source);
    assert!(matches!(
        result,
        Err(AssemblerError::InstructionOutsideKernel { .. })
    ));
}

#[test]
fn test_integration_error_unknown_instruction() {
    let source = r#"
.kernel test
    unknown_op r0, r1
.end
"#;

    let result = assemble_program(source);
    assert!(matches!(
        result,
        Err(AssemblerError::UnknownInstruction { .. })
    ));
}

#[test]
fn test_integration_register_warning() {
    let source = r#"
.kernel register_test
.registers 64
    nop
.end
"#;

    let result = assemble(source, "test.wave");
    assert!(result.is_ok());

    let assembled = result.unwrap();
    assert!(!assembled.warnings.is_empty());
}

#[test]
fn test_integration_bitwise_ops() {
    let source = r#"
.kernel bitwise
.registers 4
    and r0, r1, r2
    or r0, r1, r2
    xor r0, r1, r2
    not r0, r1
    shl r0, r1, r2
    shr r0, r1, r2
    sar r0, r1, r2
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_comparison_select() {
    let source = r#"
.kernel compare
.registers 4
    icmp_eq p0, r0, r1
    icmp_lt p1, r0, r1
    fcmp_lt p2, r0, r1
    select r3, p0, r0, r1
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_type_conversion() {
    let source = r#"
.kernel convert
.registers 4
    cvt_f32_i32 r0, r1
    cvt_i32_f32 r2, r3
    cvt_f32_f16 r0, r1
    cvt_f16_f32 r2, r3
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_cache_hints() {
    let source = r#"
.kernel cache_test
.registers 4
    device_load_u32.cached r0, r1
    device_load_u32.uncached r2, r3
    device_load_u32.streaming r0, r1
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_special_registers() {
    let source = r#"
.kernel special_regs
.registers 4
    mov r0, sr_thread_id_x
    mov r1, sr_workgroup_id_x
    mov r2, sr_wave_width
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_forward_label_reference() {
    let source = r#"
.kernel forward_ref
.registers 2
    call my_function
    halt
my_function:
    nop
    return
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_integration_all_special_registers() {
    let source = r#"
.kernel test_all_sr
.registers 16
    mov r0, sr_thread_id_x
    mov r1, sr_thread_id_y
    mov r2, sr_thread_id_z
    mov r3, sr_wave_id
    mov r4, sr_lane_id
    mov r5, sr_workgroup_id_x
    mov r6, sr_workgroup_id_y
    mov r7, sr_workgroup_id_z
    mov r8, sr_workgroup_size_x
    mov r9, sr_workgroup_size_y
    mov r10, sr_workgroup_size_z
    mov r11, sr_grid_size_x
    mov r12, sr_grid_size_y
    mov r13, sr_grid_size_z
    mov r14, sr_wave_width
    mov r15, sr_num_waves
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());

    let binary = result.unwrap();
    let code_offset = u32::from_le_bytes([binary[0x08], binary[0x09], binary[0x0A], binary[0x0B]]) as usize;

    for i in 0..16 {
        let inst_offset = code_offset + i * 4;
        let word = u32::from_le_bytes([
            binary[inst_offset],
            binary[inst_offset + 1],
            binary[inst_offset + 2],
            binary[inst_offset + 3],
        ]);

        let opcode = (word >> 26) & 0x3F;
        let rd = (word >> 21) & 0x1F;
        let rs1 = (word >> 16) & 0x1F;
        let modifier = (word >> 7) & 0x0F;
        let flags = word & 0x03;

        assert_eq!(opcode, 0x3F, "instruction {i}: expected Control opcode (0x3F)");
        assert_eq!(rd, i as u32, "instruction {i}: expected rd={i}");
        assert_eq!(rs1, i as u32, "instruction {i}: expected sr index {i}");
        assert_eq!(modifier, 2, "instruction {i}: expected MovSr modifier (2), got {modifier}");
        assert_eq!(flags, 2, "instruction {i}: expected MISC_OP_FLAG (2), got {flags}");
    }
}

#[test]
fn test_integration_mov_sr_explicit_syntax() {
    let source = r#"
.kernel test_mov_sr
.registers 4
    mov_sr r0, sr_thread_id_x
    mov_sr r1, sr_lane_id
    mov_sr r2, sr_wave_width
    mov_sr r3, sr_num_waves
.end
"#;

    let result = assemble_program(source);
    assert!(result.is_ok());

    let binary = result.unwrap();
    let code_offset = u32::from_le_bytes([binary[0x08], binary[0x09], binary[0x0A], binary[0x0B]]) as usize;

    for i in 0..4 {
        let inst_offset = code_offset + i * 4;
        let word = u32::from_le_bytes([
            binary[inst_offset],
            binary[inst_offset + 1],
            binary[inst_offset + 2],
            binary[inst_offset + 3],
        ]);

        let modifier = (word >> 7) & 0x0F;
        assert_eq!(modifier, 2, "instruction {i}: expected MovSr modifier (2), got {modifier}");
    }
}

#[test]
fn test_integration_special_register_encoding_correctness() {
    let test_cases = [
        ("sr_thread_id_x", 0),
        ("sr_thread_id_y", 1),
        ("sr_thread_id_z", 2),
        ("sr_wave_id", 3),
        ("sr_lane_id", 4),
        ("sr_workgroup_id_x", 5),
        ("sr_workgroup_id_y", 6),
        ("sr_workgroup_id_z", 7),
        ("sr_workgroup_size_x", 8),
        ("sr_workgroup_size_y", 9),
        ("sr_workgroup_size_z", 10),
        ("sr_grid_size_x", 11),
        ("sr_grid_size_y", 12),
        ("sr_grid_size_z", 13),
        ("sr_wave_width", 14),
        ("sr_num_waves", 15),
    ];

    for (sr_name, expected_index) in test_cases {
        let source = format!(
            r#"
.kernel test_sr
.registers 4
    mov r0, {sr_name}
.end
"#
        );

        let result = assemble_program(&source);
        assert!(result.is_ok(), "Failed to assemble mov r0, {sr_name}");

        let binary = result.unwrap();
        let code_offset = u32::from_le_bytes([binary[0x08], binary[0x09], binary[0x0A], binary[0x0B]]) as usize;

        let word = u32::from_le_bytes([
            binary[code_offset],
            binary[code_offset + 1],
            binary[code_offset + 2],
            binary[code_offset + 3],
        ]);

        let rs1 = (word >> 16) & 0x1F;
        assert_eq!(
            rs1, expected_index,
            "{sr_name} should encode to index {expected_index}, got {rs1}"
        );
    }
}
