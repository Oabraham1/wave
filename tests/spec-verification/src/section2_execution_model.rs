// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Section 2: Execution Model tests
//!
//! Verifies thread identity, wave structure, workgroup dimensions, and scheduling.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_thread_identity(),
        test_lane_id_unique(),
        test_wave_width(),
        test_workgroup_dims_1d(),
        test_workgroup_dims_3d(),
        test_grid_dims(),
        test_forward_progress(),
        test_multiple_waves_in_workgroup(),
    ]
}

/// Spec Section 2.3: Every thread has unique thread_id, lane_id, workgroup_id.
/// Each thread writes its lane_id to a unique memory location.
fn test_thread_identity() -> TestResult {
    const SOURCE: &str = r#"
; WAVE Spec Verification Test
; Section: 2.3 Thread Identity
; Claim: "Each Thread has a unique identity within the Grid."
; Method: Each thread writes its lane_id to device memory at offset (lane_id * 4).
;         Verify all values are present and unique.

.kernel test_thread_identity
.registers 4
    mov r0, sr_lane_id          ; r0 = lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; r2 = lane_id * 4 (offset)
    device_store_u32 r2, r0     ; store lane_id at offset
    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let mut passed = true;
            let mut details = String::new();

            for lane in 0..4u32 {
                let offset = (lane * 4) as usize;
                let value = read_u32(&result.device_memory, offset);
                if value != lane {
                    passed = false;
                    details = format!(
                        "Lane {} wrote {} instead of {} at offset {}",
                        lane, value, lane, offset
                    );
                    break;
                }
            }

            if passed {
                details = "All 4 threads wrote unique lane_ids correctly".to_string();
            }

            TestResult {
                name: "test_thread_identity".to_string(),
                spec_section: "2.3".to_string(),
                spec_claim: "Each Thread has a unique identity".to_string(),
                passed,
                details,
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_thread_identity".to_string(),
            spec_section: "2.3".to_string(),
            spec_claim: "Each Thread has a unique identity".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 2.2: Lane IDs within a wave are unique and sequential [0, W).
fn test_lane_id_unique() -> TestResult {
    const SOURCE: &str = r#"
; Section: 2.2 Wave Structure
; Claim: "Threads within a Wave are identified by lane index in [0, W)."

.kernel test_lane_id_unique
.registers 4
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Write lane_id + 100 to distinguish from uninitialized memory
    mov_imm r3, 100
    iadd r3, r0, r3
    device_store_u32 r2, r3
    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let mut passed = true;
            let mut seen = [false; 4];

            for lane in 0..4u32 {
                let offset = (lane * 4) as usize;
                let value = read_u32(&result.device_memory, offset);
                let expected = lane + 100;

                if value != expected {
                    passed = false;
                }

                if value >= 100 && value < 104 {
                    seen[(value - 100) as usize] = true;
                }
            }

            let all_unique = seen.iter().all(|&x| x);

            TestResult {
                name: "test_lane_id_unique".to_string(),
                spec_section: "2.2".to_string(),
                spec_claim: "Lane indices are unique in [0, W)".to_string(),
                passed: passed && all_unique,
                details: if passed && all_unique {
                    "All lane IDs unique and in range".to_string()
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
            name: "test_lane_id_unique".to_string(),
            spec_section: "2.2".to_string(),
            spec_claim: "Lane indices are unique in [0, W)".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 2.2: sr_wave_width returns the correct wave width.
fn test_wave_width() -> TestResult {
    const SOURCE: &str = r#"
; Section: 2.2 Wave
; Claim: "A Wave is exactly W Threads."

.kernel test_wave_width
.registers 4
    mov r0, sr_wave_width       ; Read wave width
    mov r1, sr_lane_id

    ; Only lane 0 writes the wave width
    mov_imm r3, 0
    icmp_eq p1, r1, r3
    @p1 mov_imm r2, 0
    @p1 device_store_u32 r2, r0

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
                name: "test_wave_width".to_string(),
                spec_section: "2.2".to_string(),
                spec_claim: "Wave width is exactly W threads".to_string(),
                passed,
                details: if passed {
                    format!("sr_wave_width = {} (correct)", wave_width)
                } else {
                    format!("sr_wave_width = {}, expected 4", wave_width)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_width".to_string(),
            spec_section: "2.2".to_string(),
            spec_claim: "Wave width is exactly W threads".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 2.2: 1D workgroup dimensions.
fn test_workgroup_dims_1d() -> TestResult {
    const SOURCE: &str = r#"
; Section: 2.2 Workgroup
; Claim: "Workgroup dimensions (x, y, z) specify a 3D block of threads."

.kernel test_workgroup_dims_1d
.registers 4
    mov r0, sr_thread_id_x
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = thread_id_x * 4
    device_store_u32 r2, r0     ; write thread_id_x
    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let mut passed = true;

            for tid in 0..8u32 {
                let offset = (tid * 4) as usize;
                let value = read_u32(&result.device_memory, offset);
                if value != tid {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_workgroup_dims_1d".to_string(),
                spec_section: "2.2".to_string(),
                spec_claim: "1D workgroup thread IDs are sequential".to_string(),
                passed,
                details: if passed {
                    "Thread IDs 0-7 written correctly".to_string()
                } else {
                    format!(
                        "Memory: {}",
                        format_memory_u32(&result.device_memory, 0, 8)
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_workgroup_dims_1d".to_string(),
            spec_section: "2.2".to_string(),
            spec_claim: "1D workgroup thread IDs are sequential".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 2.2: 3D workgroup dimensions.
fn test_workgroup_dims_3d() -> TestResult {
    const SOURCE: &str = r#"
; Section: 2.2 Workgroup
; Claim: "Workgroup dimensions (x, y, z) specify a 3D block."
; Method: Launch (2,2,1) workgroup = 4 threads. Each writes thread_id_x and thread_id_y.

.kernel test_workgroup_dims_3d
.registers 8
    mov r0, sr_thread_id_x
    mov r1, sr_thread_id_y
    mov r2, sr_workgroup_size_x     ; Should be 2

    ; Calculate linear index: y * width_x + x
    imul r3, r1, r2
    iadd r3, r3, r0                 ; r3 = linear_id

    ; Write to offset (linear_id * 8): [thread_id_x, thread_id_y]
    mov_imm r4, 3
    shl r5, r3, r4                  ; offset = linear_id * 8
    device_store_u32 r5, r0         ; write thread_id_x

    mov_imm r6, 4
    iadd r5, r5, r6                 ; offset + 4
    device_store_u32 r5, r1         ; write thread_id_y

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [2, 2, 1], &config, None) {
        Ok(result) => {
            let expected = [(0u32, 0u32), (1, 0), (0, 1), (1, 1)];
            let mut passed = true;
            let mut details = String::new();

            for (linear_id, (exp_x, exp_y)) in expected.iter().enumerate() {
                let offset = linear_id * 8;
                let x = read_u32(&result.device_memory, offset);
                let y = read_u32(&result.device_memory, offset + 4);

                if x != *exp_x || y != *exp_y {
                    passed = false;
                    details = format!(
                        "Thread {} has ({},{}) expected ({},{})",
                        linear_id, x, y, exp_x, exp_y
                    );
                    break;
                }
            }

            if passed {
                details = "All 4 threads have correct (x,y) coordinates".to_string();
            }

            TestResult {
                name: "test_workgroup_dims_3d".to_string(),
                spec_section: "2.2".to_string(),
                spec_claim: "3D workgroup dimensions work correctly".to_string(),
                passed,
                details,
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_workgroup_dims_3d".to_string(),
            spec_section: "2.2".to_string(),
            spec_claim: "3D workgroup dimensions work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 2.2: Grid is a 3D count of workgroups.
fn test_grid_dims() -> TestResult {
    const SOURCE: &str = r#"
; Section: 2.2 Grid
; Claim: "Grid is a 3D count of workgroups."
; Method: Launch 2x1x1 grid of 4-thread workgroups. Each thread writes workgroup_id_x.

.kernel test_grid_dims
.registers 8
    mov r0, sr_workgroup_id_x
    mov r1, sr_thread_id_x
    mov r2, sr_workgroup_size_x     ; 4

    ; global_id = workgroup_id * workgroup_size + thread_id
    imul r3, r0, r2
    iadd r3, r3, r1                 ; r3 = global_thread_id

    ; Write workgroup_id to offset (global_id * 4)
    mov_imm r4, 2
    shl r5, r3, r4
    device_store_u32 r5, r0

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [2, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let mut passed = true;
            let mut details = String::new();

            for i in 0..4 {
                let value = read_u32(&result.device_memory, i * 4);
                if value != 0 {
                    passed = false;
                    details = format!("Thread {} in workgroup 0 wrote {}", i, value);
                    break;
                }
            }

            if passed {
                for i in 4..8 {
                    let value = read_u32(&result.device_memory, i * 4);
                    if value != 1 {
                        passed = false;
                        details = format!("Thread {} in workgroup 1 wrote {}", i - 4, value);
                        break;
                    }
                }
            }

            if passed {
                details = "Both workgroups executed with correct IDs".to_string();
            }

            TestResult {
                name: "test_grid_dims".to_string(),
                spec_section: "2.2".to_string(),
                spec_claim: "Grid launches multiple workgroups".to_string(),
                passed,
                details,
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_grid_dims".to_string(),
            spec_section: "2.2".to_string(),
            spec_claim: "Grid launches multiple workgroups".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 2.5.6: Forward progress guarantee.
fn test_forward_progress() -> TestResult {
    const SOURCE: &str = r#"
; Section: 2.5.6 Forward Progress
; Claim: "Implementation MUST guarantee forward progress for at least one Wave."
; Method: Launch 2 waves. Each wave writes a completion flag. Verify all complete.

.kernel test_forward_progress
.registers 4
    mov r0, sr_wave_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = wave_id * 4

    ; Write completion flag (0xDEADBEEF)
    mov_imm r3, 0xDEADBEEF
    device_store_u32 r2, r3

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let wave0_flag = read_u32(&result.device_memory, 0);
            let wave1_flag = read_u32(&result.device_memory, 4);

            let passed = wave0_flag == COMPLETION_FLAG && wave1_flag == COMPLETION_FLAG;

            TestResult {
                name: "test_forward_progress".to_string(),
                spec_section: "2.5.6".to_string(),
                spec_claim: "Forward progress guarantee".to_string(),
                passed,
                details: if passed {
                    "Both waves completed successfully".to_string()
                } else {
                    format!(
                        "Wave flags: 0x{:08x}, 0x{:08x}",
                        wave0_flag, wave1_flag
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_forward_progress".to_string(),
            spec_section: "2.5.6".to_string(),
            spec_claim: "Forward progress guarantee".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 2.2: Multiple waves in a workgroup share local memory.
fn test_multiple_waves_in_workgroup() -> TestResult {
    const SOURCE: &str = r#"
; Section: 2.2 Workgroup
; Claim: "Waves within a Workgroup share Local Memory."
; Method: Wave 0 writes to local memory, barrier, wave 1 reads and writes to device memory.

.kernel test_multiple_waves
.registers 8
    mov r0, sr_wave_id
    mov r1, sr_lane_id

    ; Wave 0: write lane_id to local memory
    mov_imm r5, 0
    icmp_eq p1, r0, r5
    @p1 mov_imm r2, 2
    @p1 shl r3, r1, r2          ; offset = lane_id * 4
    @p1 local_store_u32 r3, r1

    ; Barrier - ensure wave 0 writes complete
    barrier

    ; Wave 1: read from local memory and write to device memory
    mov_imm r5, 1
    icmp_eq p2, r0, r5
    @p2 mov_imm r2, 2
    @p2 shl r3, r1, r2          ; offset = lane_id * 4
    @p2 local_load_u32 r4, r3
    @p2 device_store_u32 r3, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let mut passed = true;

            for lane in 0..4u32 {
                let offset = (lane * 4) as usize;
                let value = read_u32(&result.device_memory, offset);
                if value != lane {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_multiple_waves_in_workgroup".to_string(),
                spec_section: "2.2".to_string(),
                spec_claim: "Waves share local memory".to_string(),
                passed,
                details: if passed {
                    "Wave 1 successfully read Wave 0's local memory writes".to_string()
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
            name: "test_multiple_waves_in_workgroup".to_string(),
            spec_section: "2.2".to_string(),
            spec_claim: "Waves share local memory".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
