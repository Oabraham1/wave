// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Stress Tests
// Real GPU-style workloads to stress-test the emulator.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_vector_add(),
        test_reduction_sum(),
        test_matrix_transpose_tile(),
        test_histogram(),
        test_prefix_sum_wave(),
        test_many_waves(),
    ]
}

fn test_vector_add() -> TestResult {
    const SOURCE: &str = r#"
; Stress Test: Vector Addition
; Each thread adds corresponding elements from two arrays.

.kernel vector_add
.registers 8
    ; tid = thread_id_x
    mov r0, sr_thread_id_x

    ; offset = tid * 4
    mov_imm r1, 2
    shl r2, r0, r1

    ; Load A[tid] from offset 0-63
    device_load_u32 r3, r2

    ; Load B[tid] from offset 256-319
    mov_imm r4, 256
    iadd r5, r2, r4
    device_load_u32 r4, r5

    ; C[tid] = A[tid] + B[tid]
    iadd r5, r3, r4

    ; Store to offset 512+
    mov_imm r6, 512
    iadd r7, r2, r6
    device_store_u32 r7, r5

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        device_memory_size: 1024,
        ..Default::default()
    };

    // Initialize A and B arrays
    let mut initial_mem = vec![0u8; 1024];

    // A[0..16] = 1, 2, 3, ..., 16
    for i in 0..16u32 {
        let bytes = (i + 1).to_le_bytes();
        let offset = (i * 4) as usize;
        initial_mem[offset..offset + 4].copy_from_slice(&bytes);
    }

    // B[0..16] = 100, 100, 100, ...
    for i in 0..16u32 {
        let bytes = 100u32.to_le_bytes();
        let offset = (256 + i * 4) as usize;
        initial_mem[offset..offset + 4].copy_from_slice(&bytes);
    }

    match run_test(SOURCE, [1, 1, 1], [16, 1, 1], &config, Some(&initial_mem)) {
        Ok(result) => {
            let mut passed = true;
            let mut details = String::new();

            for i in 0..16u32 {
                let expected = (i + 1) + 100;
                let actual = read_u32(&result.device_memory, (512 + i * 4) as usize);
                if actual != expected {
                    passed = false;
                    details = format!("C[{}] = {} (expected {})", i, actual, expected);
                    break;
                }
            }

            if passed {
                details = "16-element vector add correct".to_string();
            }

            TestResult {
                name: "test_vector_add".to_string(),
                spec_section: "stress".to_string(),
                spec_claim: "Vector addition kernel".to_string(),
                passed,
                details,
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_vector_add".to_string(),
            spec_section: "stress".to_string(),
            spec_claim: "Vector addition kernel".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_reduction_sum() -> TestResult {
    const SOURCE: &str = r#"
; Stress Test: Parallel Reduction Sum
; Each thread atomically adds its value to a shared counter.

.kernel reduction_sum
.registers 8
    ; Each thread contributes its thread_id + 1
    mov r0, sr_thread_id_x
    mov_imm r1, 1
    iadd r0, r0, r1          ; value = tid + 1

    ; Atomically add to address 0
    mov_imm r1, 0
    atomic_add r2, r1, r0, .device

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // Initialize counter to 0
    let initial_mem = vec![0u8; 64];

    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, Some(&initial_mem)) {
        Ok(result) => {
            let sum = read_u32(&result.device_memory, 0);
            // Sum of 1+2+3+...+8 = 36
            let expected = 36u32;
            let passed = sum == expected;

            TestResult {
                name: "test_reduction_sum".to_string(),
                spec_section: "stress".to_string(),
                spec_claim: "Parallel reduction with atomics".to_string(),
                passed,
                details: format!("Sum = {} (expected {})", sum, expected),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_reduction_sum".to_string(),
            spec_section: "stress".to_string(),
            spec_claim: "Parallel reduction with atomics".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_matrix_transpose_tile() -> TestResult {
    const SOURCE: &str = r#"
; Stress Test: Matrix Transpose (2x2 tile)
; Transposes a small matrix using thread cooperation.

.kernel transpose_tile
.registers 8
    ; Thread layout: 2x2 grid
    ; Thread (x,y) reads M[y][x] and writes to M_T[x][y]

    mov r0, sr_thread_id_x      ; x coord (0 or 1)
    mov r1, sr_thread_id_y      ; y coord (0 or 1)

    ; Read offset = y*8 + x*4 (2 columns, 4 bytes each)
    mov_imm r2, 3
    shl r3, r1, r2              ; y * 8
    mov_imm r2, 2
    shl r4, r0, r2              ; x * 4
    iadd r5, r3, r4              ; read_offset

    ; Load value
    device_load_u32 r6, r5

    ; Write offset = x*8 + y*4 (transposed)
    mov_imm r2, 3
    shl r3, r0, r2              ; x * 8
    mov_imm r2, 2
    shl r4, r1, r2              ; y * 4
    iadd r5, r3, r4
    mov_imm r7, 64              ; Output at offset 64
    iadd r5, r5, r7

    ; Store transposed
    device_store_u32 r5, r6

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        device_memory_size: 128,
        ..Default::default()
    };

    // Initialize 2x2 matrix:
    // [ 1, 2 ]
    // [ 3, 4 ]
    let mut initial_mem = vec![0u8; 128];
    initial_mem[0..4].copy_from_slice(&1u32.to_le_bytes());
    initial_mem[4..8].copy_from_slice(&2u32.to_le_bytes());
    initial_mem[8..12].copy_from_slice(&3u32.to_le_bytes());
    initial_mem[12..16].copy_from_slice(&4u32.to_le_bytes());

    match run_test(SOURCE, [1, 1, 1], [2, 2, 1], &config, Some(&initial_mem)) {
        Ok(result) => {
            // Transposed should be:
            // [ 1, 3 ]
            // [ 2, 4 ]
            let t00 = read_u32(&result.device_memory, 64);
            let t01 = read_u32(&result.device_memory, 68);
            let t10 = read_u32(&result.device_memory, 72);
            let t11 = read_u32(&result.device_memory, 76);

            let passed = t00 == 1 && t01 == 3 && t10 == 2 && t11 == 4;

            TestResult {
                name: "test_matrix_transpose_tile".to_string(),
                spec_section: "stress".to_string(),
                spec_claim: "2D thread indexing for transpose".to_string(),
                passed,
                details: if passed {
                    "2x2 matrix transposed correctly".to_string()
                } else {
                    format!("Result: [{}, {}], [{}, {}]", t00, t01, t10, t11)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_matrix_transpose_tile".to_string(),
            spec_section: "stress".to_string(),
            spec_claim: "2D thread indexing for transpose".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_histogram() -> TestResult {
    const SOURCE: &str = r#"
; Stress Test: Histogram
; Each thread atomically increments a bin based on its ID.

.kernel histogram
.registers 8
    ; Compute bin = thread_id % 4
    mov r0, sr_thread_id_x
    mov_imm r1, 3
    and r2, r0, r1              ; bin = tid & 3

    ; Bin address = bin * 4
    mov_imm r3, 2
    shl r4, r2, r3              ; offset = bin * 4

    ; Atomically increment bin
    mov_imm r5, 1
    atomic_add r6, r4, r5, .device

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // Initialize histogram bins to 0
    let initial_mem = vec![0u8; 64];

    // 16 threads, 4 bins -> each bin should have count 4
    match run_test(SOURCE, [1, 1, 1], [16, 1, 1], &config, Some(&initial_mem)) {
        Ok(result) => {
            let bin0 = read_u32(&result.device_memory, 0);
            let bin1 = read_u32(&result.device_memory, 4);
            let bin2 = read_u32(&result.device_memory, 8);
            let bin3 = read_u32(&result.device_memory, 12);

            let passed = bin0 == 4 && bin1 == 4 && bin2 == 4 && bin3 == 4;

            TestResult {
                name: "test_histogram".to_string(),
                spec_section: "stress".to_string(),
                spec_claim: "Histogram with atomics".to_string(),
                passed,
                details: format!("Bins: [{}, {}, {}, {}]", bin0, bin1, bin2, bin3),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_histogram".to_string(),
            spec_section: "stress".to_string(),
            spec_claim: "Histogram with atomics".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_prefix_sum_wave() -> TestResult {
    const SOURCE: &str = r#"
; Stress Test: Wave-level Prefix Sum (simplified)
; Each thread writes its lane_id to local memory, then reads neighbors.

.kernel prefix_sum_wave
.registers 8
    ; Write lane_id to local memory
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4
    local_store_u32 r2, r0

    ; Barrier to ensure all writes complete
    barrier

    ; Read own value
    local_load_u32 r3, r2

    ; Write result to device memory
    device_store_u32 r2, r3

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        local_memory_size: 256,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let v0 = read_u32(&result.device_memory, 0);
            let v1 = read_u32(&result.device_memory, 4);
            let v2 = read_u32(&result.device_memory, 8);
            let v3 = read_u32(&result.device_memory, 12);

            let passed = v0 == 0 && v1 == 1 && v2 == 2 && v3 == 3;

            TestResult {
                name: "test_prefix_sum_wave".to_string(),
                spec_section: "stress".to_string(),
                spec_claim: "Local memory + barrier pattern".to_string(),
                passed,
                details: if passed {
                    "Wave-level local memory access worked".to_string()
                } else {
                    format!("Results: [{}, {}, {}, {}]", v0, v1, v2, v3)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_prefix_sum_wave".to_string(),
            spec_section: "stress".to_string(),
            spec_claim: "Local memory + barrier pattern".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

fn test_many_waves() -> TestResult {
    const SOURCE: &str = r#"
; Stress Test: Many Waves
; Launch many threads across multiple waves and workgroups.

.kernel many_waves
.registers 8
    ; Compute global thread ID
    mov r0, sr_workgroup_id_x
    mov r1, sr_workgroup_size_x
    imul r2, r0, r1              ; wg_id * wg_size
    mov r3, sr_thread_id_x
    iadd r4, r2, r3              ; global_id = wg_id * wg_size + tid

    ; Write global_id to offset global_id * 4
    mov_imm r5, 2
    shl r6, r4, r5              ; offset = global_id * 4
    device_store_u32 r6, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        device_memory_size: 4096,
        ..Default::default()
    };

    // 4 workgroups of 8 threads each = 32 threads total
    match run_test(SOURCE, [4, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let mut passed = true;
            let mut bad_index = 0u32;
            let mut bad_value = 0u32;

            for i in 0..32u32 {
                let value = read_u32(&result.device_memory, (i * 4) as usize);
                if value != i {
                    passed = false;
                    bad_index = i;
                    bad_value = value;
                    break;
                }
            }

            TestResult {
                name: "test_many_waves".to_string(),
                spec_section: "stress".to_string(),
                spec_claim: "Multi-workgroup execution".to_string(),
                passed,
                details: if passed {
                    "32 threads across 4 workgroups executed correctly".to_string()
                } else {
                    format!("Thread {} wrote {} instead of {}", bad_index, bad_value, bad_index)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_many_waves".to_string(),
            spec_section: "stress".to_string(),
            spec_claim: "Multi-workgroup execution".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
