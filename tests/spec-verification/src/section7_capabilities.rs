// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Section 7: Capabilities tests
// Verifies capability queries and optional feature detection.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_query_wave_width(),
        test_query_max_registers(),
        test_query_local_memory_size(),
        test_query_max_workgroup_size(),
        test_cap_f64_present(),
    ]
}

fn test_query_wave_width() -> TestResult {
    const SOURCE: &str = r#"
; Section: 7.1 Capabilities
; Claim: "WAVE_WIDTH returns the configured wave width."

.kernel test_query_wave_width
.registers 4
    mov r0, sr_wave_width
    mov_imm r1, 0
    device_store_u32 r1, r0
    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let wave_width = read_u32(&result.device_memory, 0);
            let passed = wave_width == 4;

            TestResult {
                name: "test_query_wave_width".to_string(),
                spec_section: "7.1".to_string(),
                spec_claim: "sr_wave_width returns correct value".to_string(),
                passed,
                details: format!("sr_wave_width = {} (expected 4)", wave_width),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_query_wave_width".to_string(),
            spec_section: "7.1".to_string(),
            spec_claim: "sr_wave_width returns correct value".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_query_max_registers() -> TestResult {
    const SOURCE: &str = r#"
; Section: 7.1 Capabilities
; Claim: "MAX_REGISTERS >= 32" (assembler currently supports 32)

.kernel test_query_max_registers
.registers 32
    ; Use register r31 to prove 32 registers exist
    mov_imm r31, 0xCAFEBABE
    mov_imm r0, 0
    device_store_u32 r0, r31
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0xCAFEBABE;

            TestResult {
                name: "test_query_max_registers".to_string(),
                spec_section: "7.1".to_string(),
                spec_claim: "At least 32 registers available".to_string(),
                passed,
                details: if passed {
                    "r31 accessible and works".to_string()
                } else {
                    format!("r31 value = 0x{:08x}", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_query_max_registers".to_string(),
            spec_section: "7.1".to_string(),
            spec_claim: "At least 32 registers available".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_query_local_memory_size() -> TestResult {
    const SOURCE: &str = r#"
; Section: 7.1 Capabilities
; Claim: "LOCAL_MEMORY_SIZE >= 16384"

.kernel test_query_local_memory_size
.registers 4
    ; Write to offset 16380 (last word before 16384)
    mov_imm r0, 0x12345678
    mov_imm r1, 16380
    local_store_u32 r1, r0

    ; Read back
    local_load_u32 r2, r1

    ; Write to device memory
    mov_imm r3, 0
    device_store_u32 r3, r2
    halt
.end
"#;

    let config = EmulatorConfig {
        local_memory_size: 16384,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0x12345678;

            TestResult {
                name: "test_query_local_memory_size".to_string(),
                spec_section: "7.1".to_string(),
                spec_claim: "Local memory >= 16384 bytes".to_string(),
                passed,
                details: if passed {
                    "16384 bytes of local memory accessible".to_string()
                } else {
                    format!("Read back 0x{:08x}, expected 0x12345678", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_query_local_memory_size".to_string(),
            spec_section: "7.1".to_string(),
            spec_claim: "Local memory >= 16384 bytes".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_query_max_workgroup_size() -> TestResult {
    const SOURCE: &str = r#"
; Section: 7.1 Capabilities
; Claim: "MAX_WORKGROUP_SIZE >= 256"

.kernel test_query_max_workgroup_size
.registers 4
    mov r0, sr_thread_id_x
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = thread_id * 4
    device_store_u32 r2, r0
    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 32,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [256, 1, 1], &config, None) {
        Ok(result) => {
            let mut passed = true;
            for i in 0..256u32 {
                let value = read_u32(&result.device_memory, (i * 4) as usize);
                if value != i {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_query_max_workgroup_size".to_string(),
                spec_section: "7.1".to_string(),
                spec_claim: "Workgroup size >= 256 supported".to_string(),
                passed,
                details: if passed {
                    "256 threads executed correctly".to_string()
                } else {
                    "Not all 256 threads wrote correct values".to_string()
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_query_max_workgroup_size".to_string(),
            spec_section: "7.1".to_string(),
            spec_claim: "Workgroup size >= 256 supported".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_cap_f64_present() -> TestResult {
    const SOURCE: &str = r#"
; Section: 7.2 Optional Capabilities
; Claim: "CAP_F64 enables 64-bit floating point."

.kernel test_cap_f64_present
.registers 8
    ; Load F64 value 2.0 into r0:r1
    mov_imm r0, 0x00000000      ; Low bits of 2.0
    mov_imm r1, 0x40000000      ; High bits of 2.0

    ; Load F64 value 3.0 into r2:r3
    mov_imm r2, 0x00000000      ; Low bits of 3.0
    mov_imm r3, 0x40080000      ; High bits of 3.0

    ; Add them: result in r4:r5 should be 5.0
    dadd r4, r0, r2

    ; Store result
    mov_imm r6, 0
    device_store_u32 r6, r4     ; Low bits
    mov_imm r6, 4
    device_store_u32 r6, r5     ; High bits

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let lo = read_u32(&result.device_memory, 0);
            let hi = read_u32(&result.device_memory, 4);
            let value = f64::from_bits((hi as u64) << 32 | lo as u64);

            let passed = (value - 5.0).abs() < 0.0001;

            TestResult {
                name: "test_cap_f64_present".to_string(),
                spec_section: "7.2".to_string(),
                spec_claim: "F64 operations work when enabled".to_string(),
                passed,
                details: format!("dadd(2.0, 3.0) = {} (expected 5.0)", value),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_cap_f64_present".to_string(),
            spec_section: "7.2".to_string(),
            spec_claim: "F64 operations work when enabled".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
