// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Section 6: Instruction tests
// Verifies arithmetic, bitwise, compare, convert, and memory instructions.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_iadd(),
        test_isub(),
        test_imul(),
        test_idiv_imod(),
        test_ineg_iabs(),
        test_imin_imax(),
        test_and_or_xor_not(),
        test_shl_shr_sar(),
        test_fadd_fsub(),
        test_fmul_fdiv(),
        test_fneg_fabs(),
        test_fmin_fmax(),
        test_fsqrt(),
        test_icmp_eq_ne(),
        test_icmp_lt_le_gt_ge(),
        test_fcmp(),
        test_select(),
        test_cvt_f32_i32(),
        test_cvt_i32_f32(),
        test_mov(),
        test_nop(),
    ]
}

/// Integer addition.
fn test_iadd() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.1 Integer Arithmetic
; Claim: "iadd rd, rs1, rs2: rd = rs1 + rs2"

.kernel test_iadd
.registers 8
    ; Test: 5 + 7 = 12
    mov_imm r0, 5
    mov_imm r1, 7
    iadd r2, r0, r1

    mov_imm r3, 0
    device_store_u32 r3, r2

    ; Test: overflow wraps
    mov_imm r0, 0xFFFFFFFF
    mov_imm r1, 2
    iadd r2, r0, r1             ; 0xFFFFFFFF + 2 = 0x00000001

    mov_imm r3, 4
    device_store_u32 r3, r2

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let sum = read_u32(&result.device_memory, 0);
            let overflow = read_u32(&result.device_memory, 4);

            let passed = sum == 12 && overflow == 1;

            TestResult {
                name: "test_iadd".to_string(),
                spec_section: "6.1".to_string(),
                spec_claim: "iadd performs 32-bit addition with wrap".to_string(),
                passed,
                details: format!("5+7={}, 0xFFFFFFFF+2=0x{:x}", sum, overflow),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_iadd".to_string(),
            spec_section: "6.1".to_string(),
            spec_claim: "iadd performs 32-bit addition with wrap".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Integer subtraction.
fn test_isub() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.1 Integer Arithmetic
; Claim: "isub rd, rs1, rs2: rd = rs1 - rs2"

.kernel test_isub
.registers 8
    mov_imm r0, 10
    mov_imm r1, 3
    isub r2, r0, r1             ; 10 - 3 = 7

    mov_imm r3, 0
    device_store_u32 r3, r2

    ; Test underflow wraps
    mov_imm r0, 0
    mov_imm r1, 1
    isub r2, r0, r1             ; 0 - 1 = 0xFFFFFFFF

    mov_imm r3, 4
    device_store_u32 r3, r2

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let diff = read_u32(&result.device_memory, 0);
            let underflow = read_u32(&result.device_memory, 4);

            let passed = diff == 7 && underflow == 0xFFFFFFFF;

            TestResult {
                name: "test_isub".to_string(),
                spec_section: "6.1".to_string(),
                spec_claim: "isub performs 32-bit subtraction with wrap".to_string(),
                passed,
                details: format!("10-3={}, 0-1=0x{:x}", diff, underflow),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_isub".to_string(),
            spec_section: "6.1".to_string(),
            spec_claim: "isub performs 32-bit subtraction with wrap".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Integer multiplication.
fn test_imul() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.1 Integer Arithmetic
; Claim: "imul rd, rs1, rs2: rd = (rs1 * rs2) & 0xFFFFFFFF"

.kernel test_imul
.registers 8
    mov_imm r0, 6
    mov_imm r1, 7
    imul r2, r0, r1             ; 6 * 7 = 42

    mov_imm r3, 0
    device_store_u32 r3, r2

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let product = read_u32(&result.device_memory, 0);
            let passed = product == 42;

            TestResult {
                name: "test_imul".to_string(),
                spec_section: "6.1".to_string(),
                spec_claim: "imul performs 32-bit multiplication".to_string(),
                passed,
                details: format!("6*7={}", product),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_imul".to_string(),
            spec_section: "6.1".to_string(),
            spec_claim: "imul performs 32-bit multiplication".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Integer division and modulo.
fn test_idiv_imod() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.1 Integer Arithmetic
; Claim: "idiv/imod perform signed integer division and modulo."

.kernel test_idiv_imod
.registers 8
    mov_imm r0, 17
    mov_imm r1, 5
    idiv r2, r0, r1             ; 17 / 5 = 3
    imod r3, r0, r1             ; 17 % 5 = 2

    mov_imm r4, 0
    device_store_u32 r4, r2

    mov_imm r4, 4
    device_store_u32 r4, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let quotient = read_u32(&result.device_memory, 0);
            let remainder = read_u32(&result.device_memory, 4);

            let passed = quotient == 3 && remainder == 2;

            TestResult {
                name: "test_idiv_imod".to_string(),
                spec_section: "6.1".to_string(),
                spec_claim: "idiv/imod work correctly".to_string(),
                passed,
                details: format!("17/5={}, 17%5={}", quotient, remainder),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_idiv_imod".to_string(),
            spec_section: "6.1".to_string(),
            spec_claim: "idiv/imod work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Integer negation and absolute value.
fn test_ineg_iabs() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.1 Integer Arithmetic
; Claim: "ineg/iabs compute negation and absolute value."

.kernel test_ineg_iabs
.registers 8
    mov_imm r0, 5
    ineg r1, r0                 ; -5 = 0xFFFFFFFB

    mov_imm r2, 0
    device_store_u32 r2, r1

    ; iabs of negative number
    iabs r3, r1                 ; |-5| = 5

    mov_imm r2, 4
    device_store_u32 r2, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let neg = read_u32(&result.device_memory, 0);
            let abs = read_u32(&result.device_memory, 4);

            let passed = neg == 0xFFFFFFFB && abs == 5;

            TestResult {
                name: "test_ineg_iabs".to_string(),
                spec_section: "6.1".to_string(),
                spec_claim: "ineg/iabs work correctly".to_string(),
                passed,
                details: format!("-5=0x{:x}, |-5|={}", neg, abs),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_ineg_iabs".to_string(),
            spec_section: "6.1".to_string(),
            spec_claim: "ineg/iabs work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Integer min/max.
fn test_imin_imax() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.1 Integer Arithmetic
; Claim: "imin/imax compute signed minimum/maximum."

.kernel test_imin_imax
.registers 8
    mov_imm r0, 5
    mov_imm r1, 10
    imin r2, r0, r1             ; min(5, 10) = 5
    imax r3, r0, r1             ; max(5, 10) = 10

    mov_imm r4, 0
    device_store_u32 r4, r2

    mov_imm r4, 4
    device_store_u32 r4, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let min_val = read_u32(&result.device_memory, 0);
            let max_val = read_u32(&result.device_memory, 4);

            let passed = min_val == 5 && max_val == 10;

            TestResult {
                name: "test_imin_imax".to_string(),
                spec_section: "6.1".to_string(),
                spec_claim: "imin/imax work correctly".to_string(),
                passed,
                details: format!("min(5,10)={}, max(5,10)={}", min_val, max_val),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_imin_imax".to_string(),
            spec_section: "6.1".to_string(),
            spec_claim: "imin/imax work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Bitwise operations.
fn test_and_or_xor_not() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.5 Bitwise Operations
; Claim: "and/or/xor/not perform bitwise operations."

.kernel test_and_or_xor_not
.registers 8
    mov_imm r0, 0xFF00FF00
    mov_imm r1, 0x0FF00FF0

    and r2, r0, r1              ; 0x0F000F00
    or r3, r0, r1               ; 0xFFF0FFF0
    xor r4, r0, r1              ; 0xF0F0F0F0
    not r5, r0                  ; 0x00FF00FF

    mov_imm r6, 0
    device_store_u32 r6, r2

    mov_imm r6, 4
    device_store_u32 r6, r3

    mov_imm r6, 8
    device_store_u32 r6, r4

    mov_imm r6, 12
    device_store_u32 r6, r5

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let and_val = read_u32(&result.device_memory, 0);
            let or_val = read_u32(&result.device_memory, 4);
            let xor_val = read_u32(&result.device_memory, 8);
            let not_val = read_u32(&result.device_memory, 12);

            let passed = and_val == 0x0F000F00
                && or_val == 0xFFF0FFF0
                && xor_val == 0xF0F0F0F0
                && not_val == 0x00FF00FF;

            TestResult {
                name: "test_and_or_xor_not".to_string(),
                spec_section: "6.5".to_string(),
                spec_claim: "Bitwise operations work correctly".to_string(),
                passed,
                details: format!(
                    "and=0x{:x}, or=0x{:x}, xor=0x{:x}, not=0x{:x}",
                    and_val, or_val, xor_val, not_val
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_and_or_xor_not".to_string(),
            spec_section: "6.5".to_string(),
            spec_claim: "Bitwise operations work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Shift operations.
fn test_shl_shr_sar() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.5 Bitwise Operations
; Claim: "shl/shr/sar perform shift operations."

.kernel test_shl_shr_sar
.registers 8
    mov_imm r0, 0x00000001
    mov_imm r1, 4
    shl r2, r0, r1              ; 1 << 4 = 16

    mov_imm r0, 0x00000100
    shr r3, r0, r1              ; 0x100 >> 4 = 16

    mov_imm r0, 0x80000000      ; Negative in signed
    sar r4, r0, r1              ; Sign-extended shift

    mov_imm r5, 0
    device_store_u32 r5, r2

    mov_imm r5, 4
    device_store_u32 r5, r3

    mov_imm r5, 8
    device_store_u32 r5, r4

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let shl_val = read_u32(&result.device_memory, 0);
            let shr_val = read_u32(&result.device_memory, 4);
            let sar_val = read_u32(&result.device_memory, 8);

            // sar with 0x80000000 >> 4 should be 0xF8000000 (sign extended)
            let passed = shl_val == 16 && shr_val == 16 && sar_val == 0xF8000000;

            TestResult {
                name: "test_shl_shr_sar".to_string(),
                spec_section: "6.5".to_string(),
                spec_claim: "Shift operations work correctly".to_string(),
                passed,
                details: format!(
                    "shl=0x{:x}, shr=0x{:x}, sar=0x{:x}",
                    shl_val, shr_val, sar_val
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_shl_shr_sar".to_string(),
            spec_section: "6.5".to_string(),
            spec_claim: "Shift operations work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Float addition and subtraction.
fn test_fadd_fsub() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.4 Float Arithmetic
; Claim: "fadd/fsub perform IEEE 754 single-precision operations."

.kernel test_fadd_fsub
.registers 8
    ; 1.5 + 2.5 = 4.0
    mov_imm r0, 0x3FC00000      ; 1.5f
    mov_imm r1, 0x40200000      ; 2.5f
    fadd r2, r0, r1

    mov_imm r3, 0
    device_store_u32 r3, r2

    ; 5.0 - 2.0 = 3.0
    mov_imm r0, 0x40A00000      ; 5.0f
    mov_imm r1, 0x40000000      ; 2.0f
    fsub r2, r0, r1

    mov_imm r3, 4
    device_store_u32 r3, r2

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let add_result = read_f32(&result.device_memory, 0);
            let sub_result = read_f32(&result.device_memory, 4);

            let passed = (add_result - 4.0).abs() < 0.001 && (sub_result - 3.0).abs() < 0.001;

            TestResult {
                name: "test_fadd_fsub".to_string(),
                spec_section: "6.4".to_string(),
                spec_claim: "fadd/fsub work correctly".to_string(),
                passed,
                details: format!("1.5+2.5={}, 5.0-2.0={}", add_result, sub_result),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_fadd_fsub".to_string(),
            spec_section: "6.4".to_string(),
            spec_claim: "fadd/fsub work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Float multiplication and division.
fn test_fmul_fdiv() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.4 Float Arithmetic
; Claim: "fmul/fdiv perform multiplication and division."

.kernel test_fmul_fdiv
.registers 8
    ; 3.0 * 4.0 = 12.0
    mov_imm r0, 0x40400000      ; 3.0f
    mov_imm r1, 0x40800000      ; 4.0f
    fmul r2, r0, r1

    mov_imm r3, 0
    device_store_u32 r3, r2

    ; 10.0 / 2.0 = 5.0
    mov_imm r0, 0x41200000      ; 10.0f
    mov_imm r1, 0x40000000      ; 2.0f
    fdiv r2, r0, r1

    mov_imm r3, 4
    device_store_u32 r3, r2

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let mul_result = read_f32(&result.device_memory, 0);
            let div_result = read_f32(&result.device_memory, 4);

            let passed = (mul_result - 12.0).abs() < 0.001 && (div_result - 5.0).abs() < 0.001;

            TestResult {
                name: "test_fmul_fdiv".to_string(),
                spec_section: "6.4".to_string(),
                spec_claim: "fmul/fdiv work correctly".to_string(),
                passed,
                details: format!("3.0*4.0={}, 10.0/2.0={}", mul_result, div_result),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_fmul_fdiv".to_string(),
            spec_section: "6.4".to_string(),
            spec_claim: "fmul/fdiv work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Float negation and absolute value.
fn test_fneg_fabs() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.4 Float Arithmetic
; Claim: "fneg/fabs compute negation and absolute value."

.kernel test_fneg_fabs
.registers 8
    mov_imm r0, 0x40400000      ; 3.0f
    fneg r1, r0                 ; -3.0f

    mov_imm r2, 0
    device_store_u32 r2, r1

    fabs r3, r1                 ; |-3.0| = 3.0

    mov_imm r2, 4
    device_store_u32 r2, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let neg_result = read_f32(&result.device_memory, 0);
            let abs_result = read_f32(&result.device_memory, 4);

            let passed = (neg_result + 3.0).abs() < 0.001 && (abs_result - 3.0).abs() < 0.001;

            TestResult {
                name: "test_fneg_fabs".to_string(),
                spec_section: "6.4".to_string(),
                spec_claim: "fneg/fabs work correctly".to_string(),
                passed,
                details: format!("-3.0={}, |-3.0|={}", neg_result, abs_result),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_fneg_fabs".to_string(),
            spec_section: "6.4".to_string(),
            spec_claim: "fneg/fabs work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Float min/max.
fn test_fmin_fmax() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.4 Float Arithmetic
; Claim: "fmin/fmax compute minimum/maximum."

.kernel test_fmin_fmax
.registers 8
    mov_imm r0, 0x40400000      ; 3.0f
    mov_imm r1, 0x40A00000      ; 5.0f

    fmin r2, r0, r1             ; min(3.0, 5.0) = 3.0
    fmax r3, r0, r1             ; max(3.0, 5.0) = 5.0

    mov_imm r4, 0
    device_store_u32 r4, r2

    mov_imm r4, 4
    device_store_u32 r4, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let min_result = read_f32(&result.device_memory, 0);
            let max_result = read_f32(&result.device_memory, 4);

            let passed = (min_result - 3.0).abs() < 0.001 && (max_result - 5.0).abs() < 0.001;

            TestResult {
                name: "test_fmin_fmax".to_string(),
                spec_section: "6.4".to_string(),
                spec_claim: "fmin/fmax work correctly".to_string(),
                passed,
                details: format!("min(3,5)={}, max(3,5)={}", min_result, max_result),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_fmin_fmax".to_string(),
            spec_section: "6.4".to_string(),
            spec_claim: "fmin/fmax work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Float square root.
fn test_fsqrt() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.4 Float Arithmetic
; Claim: "fsqrt computes square root."

.kernel test_fsqrt
.registers 8
    mov_imm r0, 0x41800000      ; 16.0f
    fsqrt r1, r0                ; sqrt(16.0) = 4.0

    mov_imm r2, 0
    device_store_u32 r2, r1

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let sqrt_result = read_f32(&result.device_memory, 0);

            let passed = (sqrt_result - 4.0).abs() < 0.001;

            TestResult {
                name: "test_fsqrt".to_string(),
                spec_section: "6.4".to_string(),
                spec_claim: "fsqrt works correctly".to_string(),
                passed,
                details: format!("sqrt(16.0)={}", sqrt_result),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_fsqrt".to_string(),
            spec_section: "6.4".to_string(),
            spec_claim: "fsqrt works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Integer compare equal/not equal.
fn test_icmp_eq_ne() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.7 Compare Operations
; Claim: "icmp_eq/icmp_ne set predicate based on equality."

.kernel test_icmp_eq_ne
.registers 8
    mov_imm r0, 5
    mov_imm r1, 5
    mov_imm r2, 10

    icmp_eq p1, r0, r1          ; 5 == 5 -> true
    icmp_ne p2, r0, r2          ; 5 != 10 -> true
    icmp_eq p3, r0, r2          ; 5 == 10 -> false

    mov_imm r3, 1
    mov_imm r4, 0

    ; Write p1 result
    mov_imm r5, 0
    @p1 device_store_u32 r5, r3
    @!p1 device_store_u32 r5, r4

    ; Write p2 result
    mov_imm r5, 4
    @p2 device_store_u32 r5, r3
    @!p2 device_store_u32 r5, r4

    ; Write p3 result
    mov_imm r5, 8
    @p3 device_store_u32 r5, r3
    @!p3 device_store_u32 r5, r4

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let p1 = read_u32(&result.device_memory, 0);
            let p2 = read_u32(&result.device_memory, 4);
            let p3 = read_u32(&result.device_memory, 8);

            let passed = p1 == 1 && p2 == 1 && p3 == 0;

            TestResult {
                name: "test_icmp_eq_ne".to_string(),
                spec_section: "6.7".to_string(),
                spec_claim: "icmp_eq/icmp_ne work correctly".to_string(),
                passed,
                details: format!("(5==5)={}, (5!=10)={}, (5==10)={}", p1, p2, p3),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_icmp_eq_ne".to_string(),
            spec_section: "6.7".to_string(),
            spec_claim: "icmp_eq/icmp_ne work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Integer compare less/greater.
fn test_icmp_lt_le_gt_ge() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.7 Compare Operations
; Claim: "icmp_lt/le/gt/ge compare signed integers."

.kernel test_icmp_lt_le_gt_ge
.registers 8
    mov_imm r0, 5
    mov_imm r1, 10

    icmp_lt p1, r0, r1          ; 5 < 10 -> true
    icmp_ge p2, r1, r0          ; 10 >= 5 -> true

    mov_imm r3, 1
    mov_imm r4, 0

    mov_imm r5, 0
    @p1 device_store_u32 r5, r3
    @!p1 device_store_u32 r5, r4

    mov_imm r5, 4
    @p2 device_store_u32 r5, r3
    @!p2 device_store_u32 r5, r4

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let p1 = read_u32(&result.device_memory, 0);
            let p2 = read_u32(&result.device_memory, 4);

            let passed = p1 == 1 && p2 == 1;

            TestResult {
                name: "test_icmp_lt_le_gt_ge".to_string(),
                spec_section: "6.7".to_string(),
                spec_claim: "icmp comparison operations work".to_string(),
                passed,
                details: format!("(5<10)={}, (10>=5)={}", p1, p2),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_icmp_lt_le_gt_ge".to_string(),
            spec_section: "6.7".to_string(),
            spec_claim: "icmp comparison operations work".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Float compare.
fn test_fcmp() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.7 Compare Operations
; Claim: "fcmp compares floats."

.kernel test_fcmp
.registers 8
    mov_imm r0, 0x40400000      ; 3.0f
    mov_imm r1, 0x40A00000      ; 5.0f

    fcmp_lt p1, r0, r1          ; 3.0 < 5.0 -> true

    mov_imm r3, 1
    mov_imm r4, 0

    mov_imm r5, 0
    @p1 device_store_u32 r5, r3
    @!p1 device_store_u32 r5, r4

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let p1 = read_u32(&result.device_memory, 0);
            let passed = p1 == 1;

            TestResult {
                name: "test_fcmp".to_string(),
                spec_section: "6.7".to_string(),
                spec_claim: "fcmp_lt works correctly".to_string(),
                passed,
                details: format!("(3.0<5.0)={}", p1),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_fcmp".to_string(),
            spec_section: "6.7".to_string(),
            spec_claim: "fcmp_lt works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Select instruction.
fn test_select() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.8 Select
; Claim: "select rd, ps, rs1, rs2: rd = ps ? rs1 : rs2"

.kernel test_select
.registers 8
    mov_imm r0, 1
    mov_imm r1, 0
    mov_imm r2, 100             ; true value
    mov_imm r3, 200             ; false value

    icmp_eq p1, r0, r0          ; p1 = true
    icmp_eq p2, r0, r1          ; p2 = false

    select r4, p1, r2, r3       ; r4 = 100 (true)
    select r5, p2, r2, r3       ; r5 = 200 (false)

    mov_imm r6, 0
    device_store_u32 r6, r4

    mov_imm r6, 4
    device_store_u32 r6, r5

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let select_true = read_u32(&result.device_memory, 0);
            let select_false = read_u32(&result.device_memory, 4);

            let passed = select_true == 100 && select_false == 200;

            TestResult {
                name: "test_select".to_string(),
                spec_section: "6.8".to_string(),
                spec_claim: "select works correctly".to_string(),
                passed,
                details: format!("true={}, false={}", select_true, select_false),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_select".to_string(),
            spec_section: "6.8".to_string(),
            spec_claim: "select works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Convert float to int.
fn test_cvt_f32_i32() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.9 Convert
; Claim: "cvt_f32_i32 converts signed int to float."

.kernel test_cvt_f32_i32
.registers 8
    mov_imm r0, 42
    cvt_f32_i32 r1, r0          ; float(42) = 42.0f

    mov_imm r2, 0
    device_store_u32 r2, r1

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let float_val = read_f32(&result.device_memory, 0);

            let passed = (float_val - 42.0).abs() < 0.001;

            TestResult {
                name: "test_cvt_f32_i32".to_string(),
                spec_section: "6.9".to_string(),
                spec_claim: "cvt_f32_i32 works correctly".to_string(),
                passed,
                details: format!("int(42) as float = {}", float_val),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_cvt_f32_i32".to_string(),
            spec_section: "6.9".to_string(),
            spec_claim: "cvt_f32_i32 works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Convert int to float.
fn test_cvt_i32_f32() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.9 Convert
; Claim: "cvt_i32_f32 converts float to signed int (truncation)."

.kernel test_cvt_i32_f32
.registers 8
    mov_imm r0, 0x42293333      ; 42.3f
    cvt_i32_f32 r1, r0          ; int(42.3) = 42

    mov_imm r2, 0
    device_store_u32 r2, r1

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let int_val = read_u32(&result.device_memory, 0);

            let passed = int_val == 42;

            TestResult {
                name: "test_cvt_i32_f32".to_string(),
                spec_section: "6.9".to_string(),
                spec_claim: "cvt_i32_f32 works correctly".to_string(),
                passed,
                details: format!("float(42.3) as int = {}", int_val),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_cvt_i32_f32".to_string(),
            spec_section: "6.9".to_string(),
            spec_claim: "cvt_i32_f32 works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Mov instruction.
fn test_mov() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.15 Data Movement
; Claim: "mov rd, rs copies a register."

.kernel test_mov
.registers 8
    mov_imm r0, 0x12345678
    mov r1, r0

    mov_imm r2, 0
    device_store_u32 r2, r1

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0x12345678;

            TestResult {
                name: "test_mov".to_string(),
                spec_section: "6.15".to_string(),
                spec_claim: "mov copies register correctly".to_string(),
                passed,
                details: format!("mov result = 0x{:x}", value),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_mov".to_string(),
            spec_section: "6.15".to_string(),
            spec_claim: "mov copies register correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Nop instruction.
fn test_nop() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.14 Control
; Claim: "nop does nothing."

.kernel test_nop
.registers 8
    mov_imm r0, 42
    nop
    nop
    nop
    mov_imm r1, 0
    device_store_u32 r1, r0

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 42;

            TestResult {
                name: "test_nop".to_string(),
                spec_section: "6.14".to_string(),
                spec_claim: "nop has no effect".to_string(),
                passed,
                details: format!("Value after nops = {}", value),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_nop".to_string(),
            spec_section: "6.14".to_string(),
            spec_claim: "nop has no effect".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
