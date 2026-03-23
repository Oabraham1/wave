// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Section 6: Synchronization tests
// Verifies barrier, fence, and wait operations.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_barrier_single_wave(),
        test_barrier_two_waves(),
        test_barrier_multiple(),
        test_barrier_in_loop(),
    ]
}

/// Barrier with single wave (should just continue).
fn test_barrier_single_wave() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.13 Synchronization
; Claim: "barrier synchronizes waves in a workgroup."
; Method: Single wave hits barrier and continues.

.kernel test_barrier_single_wave
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    mov_imm r3, 100
    device_store_u32 r2, r3     ; Write before barrier

    barrier                     ; All threads sync

    ; Write completion flag
    mov_imm r4, 0xDEADBEEF
    mov_imm r5, 100
    device_store_u32 r5, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let completion = read_u32(&result.device_memory, 100);
            let passed = completion == COMPLETION_FLAG;

            TestResult {
                name: "test_barrier_single_wave".to_string(),
                spec_section: "6.13".to_string(),
                spec_claim: "Barrier with single wave works".to_string(),
                passed,
                details: if passed {
                    "Single wave passed barrier".to_string()
                } else {
                    "Barrier deadlocked or crashed".to_string()
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_barrier_single_wave".to_string(),
            spec_section: "6.13".to_string(),
            spec_claim: "Barrier with single wave works".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Barrier synchronizes two waves.
fn test_barrier_two_waves() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.13 Synchronization
; Claim: "All Waves in the Workgroup MUST reach barrier before any proceeds."
; Method: Wave 0 writes, barrier, Wave 1 reads.

.kernel test_barrier_two_waves
.registers 8
    mov r0, sr_wave_id
    mov r1, sr_lane_id
    mov_imm r7, 0

    ; Wave 0, Lane 0: Write to local memory
    ; Combined check: (wave_id | lane_id) == 0
    or r5, r0, r1
    icmp_eq p3, r5, r7

    @p3 mov_imm r2, 0x12345678
    @p3 mov_imm r3, 0
    @p3 local_store_u32 r3, r2

    ; All waves hit barrier
    barrier

    ; Wave 1, Lane 0: Read from local memory and write to device
    ; Combined check: (wave_id - 1) | lane_id == 0
    mov_imm r6, 1
    isub r5, r0, r6
    or r5, r5, r1
    icmp_eq p3, r5, r7

    @p3 mov_imm r3, 0
    @p3 local_load_u32 r4, r3
    @p3 device_store_u32 r3, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // 8 threads = 2 waves
    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0x12345678;

            TestResult {
                name: "test_barrier_two_waves".to_string(),
                spec_section: "6.13".to_string(),
                spec_claim: "Barrier synchronizes two waves".to_string(),
                passed,
                details: format!(
                    "Wave 1 read 0x{:x} (expected 0x12345678)",
                    value
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_barrier_two_waves".to_string(),
            spec_section: "6.13".to_string(),
            spec_claim: "Barrier synchronizes two waves".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Multiple barriers in sequence.
fn test_barrier_multiple() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.13 Synchronization
; Claim: "Multiple barriers can be used in sequence."

.kernel test_barrier_multiple
.registers 8
    mov r0, sr_wave_id
    mov r1, sr_lane_id
    mov_imm r7, 0

    ; Wave 0, Lane 0: Write first value
    ; Combined check: (wave_id | lane_id) == 0
    or r5, r0, r1
    icmp_eq p3, r5, r7

    @p3 mov_imm r2, 111
    @p3 mov_imm r3, 0
    @p3 local_store_u32 r3, r2

    barrier

    ; Wave 1, Lane 0: Read first, write second
    ; Combined check: (wave_id - 1) | lane_id == 0
    mov_imm r6, 1
    isub r5, r0, r6
    or r5, r5, r1
    icmp_eq p3, r5, r7

    @p3 mov_imm r3, 0
    @p3 local_load_u32 r4, r3       ; Read 111
    @p3 mov_imm r5, 4
    @p3 local_store_u32 r5, r4      ; Store at offset 4

    barrier

    ; Wave 0, Lane 0: Read second value
    ; Combined check: (wave_id | lane_id) == 0
    or r5, r0, r1
    icmp_eq p3, r5, r7

    @p3 mov_imm r3, 4
    @p3 local_load_u32 r4, r3
    @p3 mov_imm r3, 0
    @p3 device_store_u32 r3, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 111;

            TestResult {
                name: "test_barrier_multiple".to_string(),
                spec_section: "6.13".to_string(),
                spec_claim: "Multiple barriers work correctly".to_string(),
                passed,
                details: format!("Final value = {} (expected 111)", value),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_barrier_multiple".to_string(),
            spec_section: "6.13".to_string(),
            spec_claim: "Multiple barriers work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Barrier inside a loop.
fn test_barrier_in_loop() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.13 Synchronization
; Claim: "Barriers can be used inside loops."
; Method: Loop 3 times, each iteration has a barrier.

.kernel test_barrier_in_loop
.registers 8
    mov r0, sr_wave_id
    mov r1, sr_lane_id
    mov_imm r7, 0               ; counter
    mov_imm r6, 3               ; loop count

    ; Compute combined predicate once
    ; (wave_id | lane_id) == 0 means wave 0, lane 0
    or r5, r0, r1

    loop
        ; Wave 0, Lane 0: increment local mem
        mov_imm r4, 0
        icmp_eq p3, r5, r4

        @p3 mov_imm r2, 0
        @p3 local_load_u32 r3, r2
        @p3 mov_imm r4, 1
        @p3 iadd r3, r3, r4
        @p3 local_store_u32 r2, r3

        barrier

        ; Increment counter
        mov_imm r4, 1
        iadd r7, r7, r4

        ; Break if counter >= 3
        icmp_ge p1, r7, r6
        break p1
    endloop

    ; Wave 0, Lane 0: write result
    mov_imm r4, 0
    icmp_eq p3, r5, r4

    @p3 mov_imm r2, 0
    @p3 local_load_u32 r3, r2
    @p3 device_store_u32 r2, r3

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 3;  // Incremented 3 times

            TestResult {
                name: "test_barrier_in_loop".to_string(),
                spec_section: "6.13".to_string(),
                spec_claim: "Barrier works inside loops".to_string(),
                passed,
                details: format!("Loop count = {} (expected 3)", value),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_barrier_in_loop".to_string(),
            spec_section: "6.13".to_string(),
            spec_claim: "Barrier works inside loops".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
