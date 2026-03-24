// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Section 3: Register Model tests
//!
//! Verifies GPR read/write, predicate registers, special registers, and register typing.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_gpr_read_write(),
        test_gpr_preserves_bits(),
        test_predicate_registers(),
        test_predicate_independence(),
        test_special_registers_thread_id(),
        test_special_registers_workgroup(),
        test_mov_imm_full_range(),
        test_register_zero_init(),
    ]
}

/// Spec Section 3.1: Registers are 32 bits, untyped.
fn test_gpr_read_write() -> TestResult {
    const SOURCE: &str = r#"
; Section: 3.1 General Purpose Registers
; Claim: "Registers are 32 bits wide."

.kernel test_gpr_read_write
.registers 8
    mov_imm r0, 0xDEADBEEF
    mov_imm r1, 0
    device_store_u32 r1, r0         ; Write r0 to device memory offset 0
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0xDEADBEEF;

            TestResult {
                name: "test_gpr_read_write".to_string(),
                spec_section: "3.1".to_string(),
                spec_claim: "Registers are 32 bits wide".to_string(),
                passed,
                details: if passed {
                    "Read 0xDEADBEEF correctly".to_string()
                } else {
                    format!("Read 0x{:08x}, expected 0xDEADBEEF", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_gpr_read_write".to_string(),
            spec_section: "3.1".to_string(),
            spec_claim: "Registers are 32 bits wide".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 3.1: Registers are untyped - bits are preserved.
fn test_gpr_preserves_bits() -> TestResult {
    const SOURCE: &str = r#"
; Section: 3.1 General Purpose Registers
; Claim: "Registers are untyped at the hardware level."
; Method: Write a bit pattern, read it back. Bits must be preserved.

.kernel test_gpr_preserves_bits
.registers 8
    ; Write various bit patterns and verify they're preserved
    mov_imm r0, 0x12345678
    mov_imm r1, 0xFFFFFFFF
    mov_imm r2, 0x00000000
    mov_imm r3, 0x80000001      ; High and low bits set

    ; Store all values
    mov_imm r4, 0
    device_store_u32 r4, r0

    mov_imm r4, 4
    device_store_u32 r4, r1

    mov_imm r4, 8
    device_store_u32 r4, r2

    mov_imm r4, 12
    device_store_u32 r4, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let expected = [0x12345678u32, 0xFFFFFFFF, 0x00000000, 0x80000001];
            let mut passed = true;
            let mut details = String::new();

            for (i, &exp) in expected.iter().enumerate() {
                let value = read_u32(&result.device_memory, i * 4);
                if value != exp {
                    passed = false;
                    details = format!("Offset {}: got 0x{:08x}, expected 0x{:08x}", i * 4, value, exp);
                    break;
                }
            }

            if passed {
                details = "All bit patterns preserved correctly".to_string();
            }

            TestResult {
                name: "test_gpr_preserves_bits".to_string(),
                spec_section: "3.1".to_string(),
                spec_claim: "Registers preserve bit patterns".to_string(),
                passed,
                details,
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_gpr_preserves_bits".to_string(),
            spec_section: "3.1".to_string(),
            spec_claim: "Registers preserve bit patterns".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 3.5: At least 4 predicate registers (p0-p3).
fn test_predicate_registers() -> TestResult {
    const SOURCE: &str = r#"
; Section: 3.5 Predicate Registers
; Claim: "At least 4 predicate registers (p0-p3)."

.kernel test_predicate_registers
.registers 8
    mov_imm r0, 5
    mov_imm r1, 3
    mov_imm r2, 5
    mov_imm r3, 10

    ; Set predicates with different comparisons
    icmp_eq p1, r0, r2          ; p1 = (5 == 5) = true
    icmp_lt p2, r1, r0          ; p2 = (3 < 5) = true
    icmp_gt p3, r1, r0          ; p3 = (3 > 5) = false

    ; Use select to test predicate values
    mov_imm r4, 1
    mov_imm r5, 0

    ; Use each predicate in a conditional store
    mov_imm r6, 0
    @p1 device_store_u32 r6, r4 ; offset 0 = 1 if p1 true

    mov_imm r6, 4
    @p2 device_store_u32 r6, r4 ; offset 4 = 1 if p2 true

    mov_imm r6, 8
    @p3 device_store_u32 r6, r4 ; offset 8 = 1 if p3 true (should be 0)

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let p1_result = read_u32(&result.device_memory, 0);  // Should be 1
            let p2_result = read_u32(&result.device_memory, 4);  // Should be 1
            let p3_result = read_u32(&result.device_memory, 8);  // Should be 0

            let passed = p1_result == 1 && p2_result == 1 && p3_result == 0;

            TestResult {
                name: "test_predicate_registers".to_string(),
                spec_section: "3.5".to_string(),
                spec_claim: "4 predicate registers work correctly".to_string(),
                passed,
                details: if passed {
                    "p1=true, p2=true, p3=false (correct)".to_string()
                } else {
                    format!(
                        "p1={}, p2={}, p3={} (expected 1, 1, 0)",
                        p1_result, p2_result, p3_result
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_predicate_registers".to_string(),
            spec_section: "3.5".to_string(),
            spec_claim: "4 predicate registers work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 3.5: Predicate registers are independent.
fn test_predicate_independence() -> TestResult {
    const SOURCE: &str = r#"
; Section: 3.5 Predicate Registers
; Claim: "Predicate registers are independent."
; Method: Set p1, modify p2, verify p1 unchanged.

.kernel test_predicate_independence
.registers 8
    mov_imm r0, 5
    mov_imm r1, 5
    mov_imm r2, 3

    ; Set p1 = true
    icmp_eq p1, r0, r1

    ; Set p2 = false (should not affect p1)
    icmp_eq p2, r0, r2

    ; Verify p1 is still true
    mov_imm r3, 0
    mov_imm r4, 1
    mov_imm r5, 0

    @p1 device_store_u32 r3, r4 ; Write 1 if p1 still true

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 1;

            TestResult {
                name: "test_predicate_independence".to_string(),
                spec_section: "3.5".to_string(),
                spec_claim: "Predicate registers are independent".to_string(),
                passed,
                details: if passed {
                    "p1 remained true after p2 was set".to_string()
                } else {
                    "p1 was incorrectly modified".to_string()
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_predicate_independence".to_string(),
            spec_section: "3.5".to_string(),
            spec_claim: "Predicate registers are independent".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 3.4: Special registers for thread identity.
fn test_special_registers_thread_id() -> TestResult {
    const SOURCE: &str = r#"
; Section: 3.4 Special Registers
; Claim: "sr_thread_id_x/y/z provide thread position within workgroup."

.kernel test_special_registers_thread_id
.registers 8
    mov r0, sr_thread_id_x
    mov r1, sr_thread_id_y
    mov r2, sr_thread_id_z
    mov r3, sr_lane_id

    ; Store all special registers
    mov_imm r4, 0
    device_store_u32 r4, r0

    mov_imm r4, 4
    device_store_u32 r4, r1

    mov_imm r4, 8
    device_store_u32 r4, r2

    mov_imm r4, 12
    device_store_u32 r4, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let tid_x = read_u32(&result.device_memory, 0);
            let tid_y = read_u32(&result.device_memory, 4);
            let tid_z = read_u32(&result.device_memory, 8);
            let lane_id = read_u32(&result.device_memory, 12);

            let passed = tid_x == 0 && tid_y == 0 && tid_z == 0 && lane_id == 0;

            TestResult {
                name: "test_special_registers_thread_id".to_string(),
                spec_section: "3.4".to_string(),
                spec_claim: "Thread ID special registers work".to_string(),
                passed,
                details: if passed {
                    format!(
                        "tid=({},{},{}), lane={}",
                        tid_x, tid_y, tid_z, lane_id
                    )
                } else {
                    format!(
                        "tid=({},{},{}), lane={} (expected all 0)",
                        tid_x, tid_y, tid_z, lane_id
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_special_registers_thread_id".to_string(),
            spec_section: "3.4".to_string(),
            spec_claim: "Thread ID special registers work".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 3.4: Special registers for workgroup identity.
fn test_special_registers_workgroup() -> TestResult {
    const SOURCE: &str = r#"
; Section: 3.4 Special Registers
; Claim: "sr_workgroup_id_x/y/z provide workgroup position in grid."
; Claim: "sr_workgroup_size_x/y/z provide workgroup dimensions."

.kernel test_special_registers_workgroup
.registers 8
    mov r0, sr_workgroup_id_x
    mov r1, sr_workgroup_size_x
    mov r2, sr_grid_size_x
    mov r3, sr_lane_id

    ; Only workgroup 0, lane 0 writes to memory
    mov_imm r5, 0
    or r6, r0, r3                ; r6 = workgroup_id | lane_id
    icmp_eq p1, r6, r5           ; p1 = (r6 == 0)

    mov_imm r4, 0
    @p1 device_store_u32 r4, r0     ; workgroup_id_x

    mov_imm r4, 4
    @p1 device_store_u32 r4, r1     ; workgroup_size_x

    mov_imm r4, 8
    @p1 device_store_u32 r4, r2     ; grid_size_x

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [2, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let wg_id = read_u32(&result.device_memory, 0);
            let wg_size = read_u32(&result.device_memory, 4);
            let grid_size = read_u32(&result.device_memory, 8);

            let passed = wg_id == 0 && wg_size == 4 && grid_size == 2;

            TestResult {
                name: "test_special_registers_workgroup".to_string(),
                spec_section: "3.4".to_string(),
                spec_claim: "Workgroup special registers work".to_string(),
                passed,
                details: format!(
                    "wg_id={}, wg_size={}, grid_size={} (expected 0, 4, 2)",
                    wg_id, wg_size, grid_size
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_special_registers_workgroup".to_string(),
            spec_section: "3.4".to_string(),
            spec_claim: "Workgroup special registers work".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Test mov_imm handles full 32-bit range.
fn test_mov_imm_full_range() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.15 Data Movement
; Claim: "mov_imm loads a 32-bit immediate into a register."

.kernel test_mov_imm_full_range
.registers 8
    mov_imm r0, 0x00000000
    mov_imm r1, 0xFFFFFFFF
    mov_imm r2, 0x80000000
    mov_imm r3, 0x7FFFFFFF

    mov_imm r4, 0
    device_store_u32 r4, r0

    mov_imm r4, 4
    device_store_u32 r4, r1

    mov_imm r4, 8
    device_store_u32 r4, r2

    mov_imm r4, 12
    device_store_u32 r4, r3

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let expected = [0x00000000u32, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF];
            let mut passed = true;

            for (i, &exp) in expected.iter().enumerate() {
                let value = read_u32(&result.device_memory, i * 4);
                if value != exp {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_mov_imm_full_range".to_string(),
                spec_section: "6.15".to_string(),
                spec_claim: "mov_imm handles full 32-bit range".to_string(),
                passed,
                details: if passed {
                    "All 32-bit values loaded correctly".to_string()
                } else {
                    format!(
                        "Memory: {}",
                        format_memory_u32(&result.device_memory, 0, 4)
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_mov_imm_full_range".to_string(),
            spec_section: "6.15".to_string(),
            spec_claim: "mov_imm handles full 32-bit range".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Test that registers are zero-initialized or that reading unwritten registers is safe.
fn test_register_zero_init() -> TestResult {
    const SOURCE: &str = r#"
; Section: 3.1 General Purpose Registers
; Note: Register initial values are implementation-defined.
; This test verifies the program runs without crash when reading uninitialized registers.

.kernel test_register_zero_init
.registers 8
    ; Read r7 without writing to it first
    mov_imm r0, 0
    device_store_u32 r0, r7     ; Store whatever r7 contains

    ; Write completion flag
    mov_imm r1, 4
    mov_imm r2, 0xDEADBEEF
    device_store_u32 r1, r2

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let completion = read_u32(&result.device_memory, 4);
            let passed = completion == COMPLETION_FLAG;

            TestResult {
                name: "test_register_zero_init".to_string(),
                spec_section: "3.1".to_string(),
                spec_claim: "Reading uninitialized registers is safe".to_string(),
                passed,
                details: if passed {
                    "Program completed without crash".to_string()
                } else {
                    "Program did not complete".to_string()
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_register_zero_init".to_string(),
            spec_section: "3.1".to_string(),
            spec_claim: "Reading uninitialized registers is safe".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
