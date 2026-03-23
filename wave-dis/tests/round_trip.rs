// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Round-trip integration tests. Assemble source, disassemble binary, reassemble
// disassembled output, and verify both binaries are byte-identical. This proves
// both assembler and disassembler are correct inverses of each other.

use wave_asm::assemble;
use wave_dis::{disassemble_wbin, DisassemblyOptions};

fn round_trip_test(source: &str) {
    let result1 = assemble(source, "test.wave").expect("first assembly failed");
    let binary1 = result1.binary;

    let options = DisassemblyOptions {
        show_offsets: false,
        show_raw: false,
        emit_directives: true,
    };

    let disassembled = disassemble_wbin(&binary1, &options).expect("disassembly failed");
    let disassembled_source = disassembled.join("\n");

    let result2 = assemble(&disassembled_source, "roundtrip.wave").expect("second assembly failed");
    let binary2 = result2.binary;

    assert_eq!(
        binary1, binary2,
        "Round-trip failed!\nOriginal source:\n{source}\n\nDisassembled:\n{disassembled_source}\n"
    );
}

#[test]
fn test_round_trip_integer_arithmetic() {
    let source = r#"
.kernel test_int
.registers 8
    iadd r0, r1, r2
    isub r3, r4, r5
    imul r6, r0, r1
    idiv r0, r3, r6
    imod r1, r2, r3
    ineg r4, r5
    iabs r6, r0
    imin r1, r2, r3
    imax r4, r5, r6
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_float_arithmetic() {
    let source = r#"
.kernel test_float
.registers 8
    fadd r0, r1, r2
    fsub r3, r4, r5
    fmul r6, r0, r1
    fdiv r0, r3, r6
    fneg r1, r2
    fabs r3, r4
    fmin r5, r6, r0
    fmax r1, r2, r3
    fsqrt r4, r5
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_bitwise() {
    let source = r#"
.kernel test_bitwise
.registers 8
    and r0, r1, r2
    or r3, r4, r5
    xor r6, r0, r1
    not r2, r3
    shl r4, r5, r6
    shr r0, r1, r2
    sar r3, r4, r5
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_memory() {
    let source = r#"
.kernel test_memory
.registers 8
    device_load_u32 r0, r1
    device_store_u32 r2, r3
    local_load_u32 r4, r5
    local_store_u32 r6, r0
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_special_registers() {
    let source = r#"
.kernel test_sr
.registers 8
    mov r0, sr_thread_id_x
    mov r1, sr_lane_id
    mov r2, sr_workgroup_id_x
    mov r3, sr_wave_width
    mov r4, sr_grid_size_x
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_mov_imm() {
    let source = r#"
.kernel test_mov
.registers 8
    mov_imm r0, 0
    mov_imm r1, 4
    mov_imm r2, 0x100
    mov_imm r3, 0xDEADBEEF
    mov r4, r0
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_compare() {
    let source = r#"
.kernel test_cmp
.registers 8
    icmp_eq p1, r0, r1
    icmp_ne p2, r2, r3
    icmp_lt p1, r4, r5
    fcmp_lt p2, r0, r1
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_convert() {
    let source = r#"
.kernel test_cvt
.registers 8
    cvt_f32_i32 r0, r1
    cvt_i32_f32 r2, r3
    cvt_f32_u32 r4, r5
    cvt_u32_f32 r6, r0
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_sync() {
    let source = r#"
.kernel test_sync
.registers 4
    barrier
    nop
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_workgroup_size() {
    let source = r#"
.kernel test_wg
.registers 4
.workgroup_size 64, 4, 1
    mov r0, sr_thread_id_x
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_multiple_kernels() {
    let source = r#"
.kernel kernel_a
.registers 4
    mov_imm r0, 1
    halt
.end

.kernel kernel_b
.registers 8
    mov_imm r0, 2
    mov_imm r1, 3
    iadd r2, r0, r1
    halt
.end
"#;
    round_trip_test(source);
}

#[test]
fn test_round_trip_vec_add() {
    let source = r#"
.kernel vec_add
.registers 8
.workgroup_size 256, 1, 1
    mov r0, sr_thread_id_x
    mov r1, sr_workgroup_id_x
    mov r2, sr_workgroup_size_x
    imul r3, r1, r2
    iadd r4, r3, r0
    mov_imm r5, 2
    shl r6, r4, r5
    device_load_u32 r0, r6
    mov_imm r7, 0x1000
    iadd r6, r6, r7
    device_load_u32 r1, r6
    iadd r2, r0, r1
    mov_imm r7, 0x2000
    mov_imm r5, 2
    shl r6, r4, r5
    iadd r6, r6, r7
    device_store_u32 r6, r2
    halt
.end
"#;
    round_trip_test(source);
}
