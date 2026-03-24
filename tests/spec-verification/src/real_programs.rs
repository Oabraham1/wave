// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Real Programs tests
//!
//! Implements and verifies actual GPU algorithms to prove the spec is viable.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_vector_add(),
        test_vector_scale(),
        test_dot_product(),
        test_parallel_reduction(),
        test_histogram(),
        test_prefix_sum(),
    ]
}

/// Vector addition: C[i] = A[i] + B[i]
fn test_vector_add() -> TestResult {
    const SOURCE: &str = r#"
; Real Program: Vector Addition
; C[i] = A[i] + B[i]
; Memory layout: A at 0, B at 64, C at 128 (16 elements each)

.kernel vector_add
.registers 8
    mov r0, sr_thread_id_x      ; Global thread ID

    ; Bounds check: if thread_id >= 16, skip
    mov_imm r1, 16
    icmp_ge p1, r0, r1
    @p1 halt

    ; Calculate addresses
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = thread_id * 4

    ; Load A[i]
    device_load_u32 r3, r2      ; A at offset 0

    ; Load B[i]
    mov_imm r4, 64
    iadd r5, r2, r4
    device_load_u32 r4, r5      ; B at offset 64

    ; C[i] = A[i] + B[i]
    iadd r5, r3, r4

    ; Store C[i]
    mov_imm r6, 128
    iadd r7, r2, r6
    device_store_u32 r7, r5     ; C at offset 128

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        device_memory_size: 1024,
        ..Default::default()
    };

    let mut init_mem = vec![0u8; 256];
    for i in 0..16u32 {
        let a_offset = (i * 4) as usize;
        let b_offset = 64 + (i * 4) as usize;
        init_mem[a_offset..a_offset + 4].copy_from_slice(&i.to_le_bytes());
        init_mem[b_offset..b_offset + 4].copy_from_slice(&100u32.to_le_bytes());
    }

    match run_test(SOURCE, [1, 1, 1], [16, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let mut passed = true;

            for i in 0..16u32 {
                let c_offset = 128 + (i * 4) as usize;
                let value = read_u32(&result.device_memory, c_offset);
                if value != i + 100 {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_vector_add".to_string(),
                spec_section: "Real Programs".to_string(),
                spec_claim: "Vector addition works".to_string(),
                passed,
                details: if passed {
                    "All 16 elements correct".to_string()
                } else {
                    format!(
                        "C = {}",
                        format_memory_u32(&result.device_memory, 128, 4)
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_vector_add".to_string(),
            spec_section: "Real Programs".to_string(),
            spec_claim: "Vector addition works".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Vector scale: B[i] = A[i] * scale
fn test_vector_scale() -> TestResult {
    const SOURCE: &str = r#"
; Real Program: Vector Scale (SAXPY simplified)
; B[i] = A[i] * 3

.kernel vector_scale
.registers 8
    mov r0, sr_thread_id_x

    mov_imm r1, 8
    icmp_ge p1, r0, r1
    @p1 halt

    ; offset = thread_id * 4
    mov_imm r1, 2
    shl r2, r0, r1

    ; Load A[i]
    device_load_u32 r3, r2

    ; Multiply by 3
    mov_imm r4, 3
    imul r5, r3, r4

    ; Store to B (offset 64)
    mov_imm r6, 64
    iadd r7, r2, r6
    device_store_u32 r7, r5

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    let mut init_mem = vec![0u8; 128];
    for i in 0..8u32 {
        let offset = (i * 4) as usize;
        init_mem[offset..offset + 4].copy_from_slice(&(i + 1).to_le_bytes());
    }

    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let mut passed = true;

            for i in 0..8u32 {
                let b_offset = 64 + (i * 4) as usize;
                let value = read_u32(&result.device_memory, b_offset);
                let expected = (i + 1) * 3;
                if value != expected {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_vector_scale".to_string(),
                spec_section: "Real Programs".to_string(),
                spec_claim: "Vector scale works".to_string(),
                passed,
                details: if passed {
                    "B = [3, 6, 9, 12, 15, 18, 21, 24]".to_string()
                } else {
                    format!(
                        "B = {}",
                        format_memory_u32(&result.device_memory, 64, 8)
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_vector_scale".to_string(),
            spec_section: "Real Programs".to_string(),
            spec_claim: "Vector scale works".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Dot product using wave reduce.
fn test_dot_product() -> TestResult {
    const SOURCE: &str = r#"
; Real Program: Dot Product
; result = sum(A[i] * B[i])

.kernel dot_product
.registers 8
    mov r0, sr_lane_id

    ; offset = lane_id * 4
    mov_imm r1, 2
    shl r2, r0, r1

    ; Load A[i] and B[i]
    device_load_u32 r3, r2      ; A at offset 0

    mov_imm r4, 16
    iadd r5, r2, r4
    device_load_u32 r4, r5      ; B at offset 16

    ; Multiply
    imul r5, r3, r4

    ; Reduce sum across wave
    wave_reduce_add r6, r5

    ; Lane 0 writes result
    mov_imm r7, 0
    icmp_eq p1, r0, r7
    @p1 mov_imm r2, 32
    @p1 device_store_u32 r2, r6

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    let mut init_mem = vec![0u8; 64];
    let a_vals = [1u32, 2, 3, 4];
    let b_vals = [5u32, 6, 7, 8];
    for (i, (&a, &b)) in a_vals.iter().zip(&b_vals).enumerate() {
        init_mem[i * 4..(i + 1) * 4].copy_from_slice(&a.to_le_bytes());
        init_mem[16 + i * 4..16 + (i + 1) * 4].copy_from_slice(&b.to_le_bytes());
    }

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let dot = read_u32(&result.device_memory, 32);
            let passed = dot == 70;

            TestResult {
                name: "test_dot_product".to_string(),
                spec_section: "Real Programs".to_string(),
                spec_claim: "Dot product works".to_string(),
                passed,
                details: format!("dot([1,2,3,4], [5,6,7,8]) = {} (expected 70)", dot),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_dot_product".to_string(),
            spec_section: "Real Programs".to_string(),
            spec_claim: "Dot product works".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Parallel reduction using wave operations.
fn test_parallel_reduction() -> TestResult {
    const SOURCE: &str = r#"
; Real Program: Parallel Reduction (sum)
; Uses wave_reduce_add

.kernel parallel_reduction
.registers 8
    mov r0, sr_lane_id

    ; Load value (each thread loads its index + 1)
    mov_imm r1, 1
    iadd r2, r0, r1             ; value = lane_id + 1

    ; Reduce across wave
    wave_reduce_add r3, r2

    ; Lane 0 writes result
    mov_imm r4, 0
    icmp_eq p1, r0, r4
    @p1 device_store_u32 r4, r3

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let sum = read_u32(&result.device_memory, 0);
            let passed = sum == 10;

            TestResult {
                name: "test_parallel_reduction".to_string(),
                spec_section: "Real Programs".to_string(),
                spec_claim: "Parallel reduction works".to_string(),
                passed,
                details: format!("sum(1,2,3,4) = {} (expected 10)", sum),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_parallel_reduction".to_string(),
            spec_section: "Real Programs".to_string(),
            spec_claim: "Parallel reduction works".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Simple histogram using atomics.
fn test_histogram() -> TestResult {
    const SOURCE: &str = r#"
; Real Program: Histogram
; Count occurrences of values 0-3 in input

.kernel histogram
.registers 8
    mov r0, sr_lane_id

    ; Load input value (use lane_id % 2 as the value)
    mov_imm r1, 1
    and r2, r0, r1              ; value = lane_id & 1 (0 or 1)

    ; Calculate histogram bin address (value * 4)
    mov_imm r3, 2
    shl r4, r2, r3              ; offset = value * 4

    ; Atomic increment
    mov_imm r5, 1
    atomic_add r6, r4, r5, .device

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    let init_mem = vec![0u8; 16];

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let bin0 = read_u32(&result.device_memory, 0);  // Even lanes (0, 2)
            let bin1 = read_u32(&result.device_memory, 4);  // Odd lanes (1, 3)

            let passed = bin0 == 2 && bin1 == 2;

            TestResult {
                name: "test_histogram".to_string(),
                spec_section: "Real Programs".to_string(),
                spec_claim: "Atomic histogram works".to_string(),
                passed,
                details: format!("histogram = [{}, {}] (expected [2, 2])", bin0, bin1),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_histogram".to_string(),
            spec_section: "Real Programs".to_string(),
            spec_claim: "Atomic histogram works".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Exclusive prefix sum.
fn test_prefix_sum() -> TestResult {
    const SOURCE: &str = r#"
; Real Program: Exclusive Prefix Sum
; output[i] = sum(input[0..i])

.kernel prefix_sum
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Load input
    device_load_u32 r3, r2

    ; Compute exclusive prefix sum
    wave_prefix_sum r4, r3

    ; Store result at offset 64
    mov_imm r5, 64
    iadd r6, r2, r5
    device_store_u32 r6, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    let mut init_mem = vec![0u8; 128];
    for i in 0..4u32 {
        let offset = (i * 4) as usize;
        init_mem[offset..offset + 4].copy_from_slice(&(i + 1).to_le_bytes());
    }

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let out0 = read_u32(&result.device_memory, 64);
            let out1 = read_u32(&result.device_memory, 68);
            let out2 = read_u32(&result.device_memory, 72);
            let out3 = read_u32(&result.device_memory, 76);

            let passed = out0 == 0 && out1 == 1 && out2 == 3 && out3 == 6;

            TestResult {
                name: "test_prefix_sum".to_string(),
                spec_section: "Real Programs".to_string(),
                spec_claim: "Prefix sum works".to_string(),
                passed,
                details: format!(
                    "prefix_sum([1,2,3,4]) = [{}, {}, {}, {}] (expected [0, 1, 3, 6])",
                    out0, out1, out2, out3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_prefix_sum".to_string(),
            spec_section: "Real Programs".to_string(),
            spec_claim: "Prefix sum works".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
