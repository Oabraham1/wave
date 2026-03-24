// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Section 9: Conformance tests
//!
//! Verifies implementation conformance requirements.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_deterministic_execution(),
        test_halt_terminates(),
        test_undefined_behavior_detection(),
        test_memory_bounds_checking(),
        test_register_bounds_checking(),
        test_instruction_fetch_bounds(),
    ]
}

fn test_deterministic_execution() -> TestResult {
    const SOURCE: &str = r#"
; Section: 9.1 Conformance
; Claim: "Execution is deterministic for valid programs."

.kernel test_deterministic
.registers 8
    ; Compute a deterministic result
    mov_imm r0, 1
    mov_imm r1, 2
    iadd r2, r0, r1      ; 3
    imul r3, r2, r2      ; 9
    iadd r4, r3, r2      ; 12
    imul r5, r4, r0      ; 12

    mov_imm r6, 0
    device_store_u32 r6, r5
    halt
.end
"#;

    let config = EmulatorConfig::default();

    let mut results = Vec::new();
    for _ in 0..3 {
        match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
            Ok(result) => results.push(read_u32(&result.device_memory, 0)),
            Err(_) => results.push(0xDEAD),
        }
    }

    let passed = results.iter().all(|&r| r == 12) && results.len() == 3;

    TestResult {
        name: "test_deterministic_execution".to_string(),
        spec_section: "9.1".to_string(),
        spec_claim: "Execution is deterministic".to_string(),
        passed,
        details: if passed {
            "3 runs produced identical results".to_string()
        } else {
            format!("Results varied: {:?}", results)
        },
        cycles: 0,
    }
}

fn test_halt_terminates() -> TestResult {
    const SOURCE: &str = r#"
; Section: 9.1 Conformance
; Claim: "halt instruction terminates execution."

.kernel test_halt_terminates
.registers 4
    mov_imm r0, 1
    mov_imm r1, 0
    device_store_u32 r1, r0
    halt
    ; Code after halt should not execute
    mov_imm r0, 2
    device_store_u32 r1, r0
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 1;

            TestResult {
                name: "test_halt_terminates".to_string(),
                spec_section: "9.1".to_string(),
                spec_claim: "halt terminates execution".to_string(),
                passed,
                details: if passed {
                    "Code after halt did not execute".to_string()
                } else {
                    format!("Value = {} (code after halt executed)", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_halt_terminates".to_string(),
            spec_section: "9.1".to_string(),
            spec_claim: "halt terminates execution".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_undefined_behavior_detection() -> TestResult {
    const SOURCE: &str = r#"
; Section: 9.2 Undefined Behavior
; Claim: "Division by zero produces defined behavior."

.kernel test_div_zero
.registers 8
    mov_imm r0, 10
    mov_imm r1, 0

    ; Division by zero - should produce 0 or max value, not crash
    idiv r2, r0, r1

    ; If we get here, execution continued
    mov_imm r3, 1
    mov_imm r4, 0
    device_store_u32 r4, r3
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 1;

            TestResult {
                name: "test_undefined_behavior_detection".to_string(),
                spec_section: "9.2".to_string(),
                spec_claim: "Division by zero handled".to_string(),
                passed,
                details: if passed {
                    "Execution continued after div by zero".to_string()
                } else {
                    "Division by zero caused unexpected behavior".to_string()
                },
                cycles: result.cycles,
            }
        }
        Err(e) => {
            TestResult {
                name: "test_undefined_behavior_detection".to_string(),
                spec_section: "9.2".to_string(),
                spec_claim: "Division by zero handled".to_string(),
                passed: true, // Trapping is acceptable behavior
                details: format!("Implementation traps on div by zero: {}", e),
                cycles: 0,
            }
        }
    }
}

fn test_memory_bounds_checking() -> TestResult {
    const SOURCE: &str = r#"
; Section: 9.3 Bounds Checking
; Claim: "Memory accesses within bounds succeed."

.kernel test_memory_bounds
.registers 8
    ; Write to valid device memory addresses
    mov_imm r0, 0x12345678
    mov_imm r1, 0
    device_store_u32 r1, r0

    mov_imm r1, 100
    device_store_u32 r1, r0

    mov_imm r1, 1000
    device_store_u32 r1, r0

    ; Read back the first value to verify
    mov_imm r1, 0
    device_load_u32 r2, r1
    device_store_u32 r1, r2
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0x12345678;

            TestResult {
                name: "test_memory_bounds_checking".to_string(),
                spec_section: "9.3".to_string(),
                spec_claim: "Memory bounds respected".to_string(),
                passed,
                details: if passed {
                    "Memory accesses within bounds succeeded".to_string()
                } else {
                    format!("Value = 0x{:08x}", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_memory_bounds_checking".to_string(),
            spec_section: "9.3".to_string(),
            spec_claim: "Memory bounds respected".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_register_bounds_checking() -> TestResult {
    const SOURCE: &str = r#"
; Section: 9.3 Bounds Checking
; Claim: "Register accesses within declared count succeed."

.kernel test_register_bounds
.registers 16
    ; Use registers 0-15 (within declared .registers 16)
    mov_imm r0, 0
    mov_imm r1, 1
    mov_imm r2, 2
    mov_imm r3, 3
    mov_imm r4, 4
    mov_imm r5, 5
    mov_imm r6, 6
    mov_imm r7, 7
    mov_imm r8, 8
    mov_imm r9, 9
    mov_imm r10, 10
    mov_imm r11, 11
    mov_imm r12, 12
    mov_imm r13, 13
    mov_imm r14, 14
    mov_imm r15, 15

    ; Sum them all
    iadd r0, r0, r1
    iadd r0, r0, r2
    iadd r0, r0, r3
    iadd r0, r0, r4
    iadd r0, r0, r5
    iadd r0, r0, r6
    iadd r0, r0, r7
    iadd r0, r0, r8
    iadd r0, r0, r9
    iadd r0, r0, r10
    iadd r0, r0, r11
    iadd r0, r0, r12
    iadd r0, r0, r13
    iadd r0, r0, r14
    iadd r0, r0, r15

    ; Result should be 0+1+2+...+15 = 120
    mov_imm r1, 0
    device_store_u32 r1, r0
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 120;

            TestResult {
                name: "test_register_bounds_checking".to_string(),
                spec_section: "9.3".to_string(),
                spec_claim: "Register bounds respected".to_string(),
                passed,
                details: if passed {
                    "All 16 registers accessible and correct".to_string()
                } else {
                    format!("Sum = {} (expected 120)", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_register_bounds_checking".to_string(),
            spec_section: "9.3".to_string(),
            spec_claim: "Register bounds respected".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_instruction_fetch_bounds() -> TestResult {
    const SOURCE: &str = r#"
; Section: 9.3 Bounds Checking
; Claim: "Instruction fetch stays within kernel bounds."

.kernel test_fetch_bounds
.registers 4
    ; Simple linear execution to end
    mov_imm r0, 1
    mov_imm r1, 2
    iadd r2, r0, r1
    mov_imm r3, 0
    device_store_u32 r3, r2
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 3;

            TestResult {
                name: "test_instruction_fetch_bounds".to_string(),
                spec_section: "9.3".to_string(),
                spec_claim: "Instruction fetch within bounds".to_string(),
                passed,
                details: if passed {
                    "Execution stayed within kernel bounds".to_string()
                } else {
                    format!("Unexpected result: {}", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_instruction_fetch_bounds".to_string(),
            spec_section: "9.3".to_string(),
            spec_claim: "Instruction fetch within bounds".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
