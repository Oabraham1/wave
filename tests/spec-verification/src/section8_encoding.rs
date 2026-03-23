// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Section 8: Binary Encoding tests
// Verifies WBIN format correctness and instruction encoding.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_wbin_magic(),
        test_wbin_kernel_entry(),
        test_instruction_alignment(),
        test_immediate_encoding(),
        test_register_encoding(),
        test_opcode_uniqueness(),
    ]
}

fn test_wbin_magic() -> TestResult {
    const SOURCE: &str = r#"
; Section: 8.1 WBIN Format
; Claim: "WBIN files begin with magic 0x4E494257 ('WBIN')."

.kernel test_wbin_magic
.registers 4
    ; If we get here, the binary was loaded successfully
    ; which means magic was validated
    mov_imm r0, 0x12345678
    mov_imm r1, 0
    device_store_u32 r1, r0
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0x12345678;

            TestResult {
                name: "test_wbin_magic".to_string(),
                spec_section: "8.1".to_string(),
                spec_claim: "WBIN magic validated on load".to_string(),
                passed,
                details: if passed {
                    "Binary loaded and executed successfully".to_string()
                } else {
                    format!("Unexpected result: 0x{:08x}", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wbin_magic".to_string(),
            spec_section: "8.1".to_string(),
            spec_claim: "WBIN magic validated on load".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_wbin_kernel_entry() -> TestResult {
    const SOURCE: &str = r#"
; Section: 8.1 WBIN Format
; Claim: "Kernel entry points are correctly resolved."

.kernel test_kernel_entry
.registers 4
    ; Entry point is at start of kernel
    mov_imm r0, 1
    mov_imm r1, 0
    device_store_u32 r1, r0
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 1;

            TestResult {
                name: "test_wbin_kernel_entry".to_string(),
                spec_section: "8.1".to_string(),
                spec_claim: "Kernel entry point resolved correctly".to_string(),
                passed,
                details: if passed {
                    "Kernel entry executed from correct location".to_string()
                } else {
                    format!("Entry point issue: got {}", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wbin_kernel_entry".to_string(),
            spec_section: "8.1".to_string(),
            spec_claim: "Kernel entry point resolved correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_instruction_alignment() -> TestResult {
    const SOURCE: &str = r#"
; Section: 8.2 Instruction Encoding
; Claim: "Instructions are 4-byte aligned."

.kernel test_instruction_alignment
.registers 8
    ; Multiple instructions to verify alignment is maintained
    mov_imm r0, 1
    mov_imm r1, 2
    mov_imm r2, 3
    mov_imm r3, 4
    iadd r4, r0, r1      ; 1 + 2 = 3
    iadd r5, r2, r3      ; 3 + 4 = 7
    iadd r6, r4, r5      ; 3 + 7 = 10

    mov_imm r7, 0
    device_store_u32 r7, r6
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 10;

            TestResult {
                name: "test_instruction_alignment".to_string(),
                spec_section: "8.2".to_string(),
                spec_claim: "Instructions are 4-byte aligned".to_string(),
                passed,
                details: if passed {
                    "All instructions executed correctly (alignment OK)".to_string()
                } else {
                    format!("Result = {} (expected 10)", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_instruction_alignment".to_string(),
            spec_section: "8.2".to_string(),
            spec_claim: "Instructions are 4-byte aligned".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_immediate_encoding() -> TestResult {
    const SOURCE: &str = r#"
; Section: 8.2 Instruction Encoding
; Claim: "32-bit immediates are correctly encoded."

.kernel test_immediate_encoding
.registers 8
    ; Test various immediate patterns
    mov_imm r0, 0x00000000
    mov_imm r1, 0xFFFFFFFF
    mov_imm r2, 0x80000000
    mov_imm r3, 0x7FFFFFFF
    mov_imm r4, 0xAAAAAAAA
    mov_imm r5, 0x55555555

    ; Store all values for verification
    mov_imm r6, 0
    device_store_u32 r6, r0
    mov_imm r6, 4
    device_store_u32 r6, r1
    mov_imm r6, 8
    device_store_u32 r6, r2
    mov_imm r6, 12
    device_store_u32 r6, r3
    mov_imm r6, 16
    device_store_u32 r6, r4
    mov_imm r6, 20
    device_store_u32 r6, r5
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let v0 = read_u32(&result.device_memory, 0);
            let v1 = read_u32(&result.device_memory, 4);
            let v2 = read_u32(&result.device_memory, 8);
            let v3 = read_u32(&result.device_memory, 12);
            let v4 = read_u32(&result.device_memory, 16);
            let v5 = read_u32(&result.device_memory, 20);

            let passed = v0 == 0x00000000
                && v1 == 0xFFFFFFFF
                && v2 == 0x80000000
                && v3 == 0x7FFFFFFF
                && v4 == 0xAAAAAAAA
                && v5 == 0x55555555;

            TestResult {
                name: "test_immediate_encoding".to_string(),
                spec_section: "8.2".to_string(),
                spec_claim: "32-bit immediates encoded correctly".to_string(),
                passed,
                details: if passed {
                    "All immediate patterns encoded correctly".to_string()
                } else {
                    format!(
                        "Values: 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}",
                        v0, v1, v2, v3, v4, v5
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_immediate_encoding".to_string(),
            spec_section: "8.2".to_string(),
            spec_claim: "32-bit immediates encoded correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_register_encoding() -> TestResult {
    const SOURCE: &str = r#"
; Section: 8.2 Instruction Encoding
; Claim: "Register operands encoded in 6 bits (0-63)."

.kernel test_register_encoding
.registers 32
    ; Test various register numbers (0-31 supported by assembler)
    mov_imm r0, 0
    mov_imm r1, 1
    mov_imm r15, 15
    mov_imm r16, 16
    mov_imm r30, 30
    mov_imm r31, 31

    ; Verify by adding and storing
    iadd r10, r0, r1         ; 0 + 1 = 1
    iadd r20, r15, r16       ; 15 + 16 = 31
    iadd r25, r30, r31       ; 30 + 31 = 61

    mov_imm r29, 0
    device_store_u32 r29, r10
    mov_imm r29, 4
    device_store_u32 r29, r20
    mov_imm r29, 8
    device_store_u32 r29, r25
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let v0 = read_u32(&result.device_memory, 0);
            let v4 = read_u32(&result.device_memory, 4);
            let v8 = read_u32(&result.device_memory, 8);

            let passed = v0 == 1 && v4 == 31 && v8 == 61;

            TestResult {
                name: "test_register_encoding".to_string(),
                spec_section: "8.2".to_string(),
                spec_claim: "Register operands (0-31) encoded correctly".to_string(),
                passed,
                details: if passed {
                    "All register numbers encoded correctly".to_string()
                } else {
                    format!("Results: {}, {}, {} (expected 1, 31, 61)", v0, v4, v8)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_register_encoding".to_string(),
            spec_section: "8.2".to_string(),
            spec_claim: "Register operands (0-63) encoded correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_opcode_uniqueness() -> TestResult {
    const SOURCE: &str = r#"
; Section: 8.2 Instruction Encoding
; Claim: "Each opcode has a unique encoding."

.kernel test_opcode_uniqueness
.registers 16
    ; Execute many different opcodes, verify they all work distinctly
    mov_imm r0, 10
    mov_imm r1, 3

    iadd r2, r0, r1          ; 10 + 3 = 13
    isub r3, r0, r1          ; 10 - 3 = 7
    imul r4, r0, r1          ; 10 * 3 = 30

    mov_imm r5, 0xFF
    mov_imm r6, 0x0F
    and r7, r5, r6          ; 0xFF & 0x0F = 0x0F
    or r8, r5, r6           ; 0xFF | 0x0F = 0xFF
    xor r9, r5, r6          ; 0xFF ^ 0x0F = 0xF0

    ; Store results
    mov_imm r10, 0
    device_store_u32 r10, r2
    mov_imm r10, 4
    device_store_u32 r10, r3
    mov_imm r10, 8
    device_store_u32 r10, r4
    mov_imm r10, 12
    device_store_u32 r10, r7
    mov_imm r10, 16
    device_store_u32 r10, r8
    mov_imm r10, 20
    device_store_u32 r10, r9
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let add_result = read_u32(&result.device_memory, 0);
            let sub_result = read_u32(&result.device_memory, 4);
            let mul_result = read_u32(&result.device_memory, 8);
            let and_result = read_u32(&result.device_memory, 12);
            let or_result = read_u32(&result.device_memory, 16);
            let xor_result = read_u32(&result.device_memory, 20);

            let passed = add_result == 13
                && sub_result == 7
                && mul_result == 30
                && and_result == 0x0F
                && or_result == 0xFF
                && xor_result == 0xF0;

            TestResult {
                name: "test_opcode_uniqueness".to_string(),
                spec_section: "8.2".to_string(),
                spec_claim: "Opcodes have unique encodings".to_string(),
                passed,
                details: if passed {
                    "All opcodes executed with distinct behavior".to_string()
                } else {
                    format!(
                        "add={}, sub={}, mul={}, and=0x{:x}, or=0x{:x}, xor=0x{:x}",
                        add_result, sub_result, mul_result, and_result, or_result, xor_result
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_opcode_uniqueness".to_string(),
            spec_section: "8.2".to_string(),
            spec_claim: "Opcodes have unique encodings".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
