// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Section 6: Wave Operations tests
//!
//! Verifies shuffle, broadcast, ballot, reduce, and vote operations.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_wave_shuffle(),
        test_wave_shuffle_xor(),
        test_wave_broadcast(),
        test_wave_ballot(),
        test_wave_any(),
        test_wave_all(),
        test_wave_reduce_add(),
        test_wave_prefix_sum(),
    ]
}

/// Wave shuffle reads from another lane.
fn test_wave_shuffle() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_shuffle reads value from another lane."
; Method: Each thread reads from lane (lane_id + 1) % 4.

.kernel test_wave_shuffle
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Set r3 = lane_id * 10 (so lane 0 has 0, lane 1 has 10, etc.)
    mov_imm r4, 10
    imul r3, r0, r4

    ; Shuffle: read from (lane_id + 1) % 4
    mov_imm r5, 1
    iadd r6, r0, r5
    mov_imm r5, 3
    and r6, r6, r5              ; r6 = (lane_id + 1) % 4

    wave_shuffle r7, r3, r6     ; r7 = value from lane r6

    device_store_u32 r2, r7

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);
            let val1 = read_u32(&result.device_memory, 4);
            let val2 = read_u32(&result.device_memory, 8);
            let val3 = read_u32(&result.device_memory, 12);

            let passed = val0 == 10 && val1 == 20 && val2 == 30 && val3 == 0;

            TestResult {
                name: "test_wave_shuffle".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_shuffle reads from specified lane".to_string(),
                passed,
                details: format!(
                    "lanes=[{}, {}, {}, {}] (expected [10, 20, 30, 0])",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_shuffle".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_shuffle reads from specified lane".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Wave shuffle XOR swaps lanes.
fn test_wave_shuffle_xor() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_shuffle_xor reads from lane (lane_id ^ mask)."

.kernel test_wave_shuffle_xor
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Each lane has its lane_id as value
    mov r3, r0

    ; XOR with 1: swaps lanes 0<->1, 2<->3
    mov_imm r4, 1
    wave_shuffle_xor r5, r3, r4

    device_store_u32 r2, r5

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);
            let val1 = read_u32(&result.device_memory, 4);
            let val2 = read_u32(&result.device_memory, 8);
            let val3 = read_u32(&result.device_memory, 12);

            let passed = val0 == 1 && val1 == 0 && val2 == 3 && val3 == 2;

            TestResult {
                name: "test_wave_shuffle_xor".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_shuffle_xor works correctly".to_string(),
                passed,
                details: format!(
                    "lanes=[{}, {}, {}, {}] (expected [1, 0, 3, 2])",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_shuffle_xor".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_shuffle_xor works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Wave broadcast reads from a single lane.
fn test_wave_broadcast() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_broadcast reads value from specified lane to all lanes."

.kernel test_wave_broadcast
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Lane 2 has the value 42
    mov_imm r3, 0
    mov_imm r4, 2
    icmp_eq p1, r0, r4
    @p1 mov_imm r3, 42

    ; Broadcast from lane 2 to all
    wave_broadcast r5, r3, r4

    device_store_u32 r2, r5

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
            for lane in 0..4 {
                let value = read_u32(&result.device_memory, lane * 4);
                if value != 42 {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_wave_broadcast".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_broadcast copies to all lanes".to_string(),
                passed,
                details: format_memory_u32(&result.device_memory, 0, 4),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_broadcast".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_broadcast copies to all lanes".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Wave ballot creates a bitmask of active threads.
fn test_wave_ballot() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_ballot returns bitmask where bit i is set if lane i's predicate is true."

.kernel test_wave_ballot
.registers 8
    mov r0, sr_lane_id

    ; Set p1 = true for even lanes
    mov_imm r1, 1
    and r2, r0, r1
    mov_imm r3, 0
    icmp_eq p1, r2, r3          ; p1 = true if lane_id & 1 == 0

    wave_ballot r4, p1          ; r4 = bitmask (0b0101 = 5 for lanes 0,2)

    ; Only lane 0 writes result
    icmp_eq p2, r0, r3
    @p2 mov_imm r5, 0
    @p2 device_store_u32 r5, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let ballot = read_u32(&result.device_memory, 0);
            let passed = ballot == 5;

            TestResult {
                name: "test_wave_ballot".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_ballot creates correct bitmask".to_string(),
                passed,
                details: format!("ballot = 0b{:04b} (expected 0b0101)", ballot),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_ballot".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_ballot creates correct bitmask".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Wave any returns true if any lane's predicate is true.
fn test_wave_any() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_any returns true if any lane's predicate is true."

.kernel test_wave_any
.registers 8
    mov r0, sr_lane_id

    ; Only lane 2 has true predicate
    mov_imm r1, 2
    icmp_eq p1, r0, r1

    wave_any p2, p1             ; p2 = true if any lane has p1 true

    ; Write result
    mov_imm r2, 0
    mov_imm r3, 1
    mov_imm r4, 0
    @p2 device_store_u32 r2, r3
    @!p2 device_store_u32 r2, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let any_result = read_u32(&result.device_memory, 0);
            let passed = any_result == 1;

            TestResult {
                name: "test_wave_any".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_any detects any true predicate".to_string(),
                passed,
                details: format!("wave_any = {} (expected 1)", any_result),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_any".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_any detects any true predicate".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Wave all returns true only if all lanes' predicates are true.
fn test_wave_all() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_all returns true only if all lanes' predicates are true."

.kernel test_wave_all
.registers 8
    mov r0, sr_lane_id

    ; Test 1: Some lanes false (lane 0 has false)
    mov_imm r1, 0
    icmp_ne p1, r0, r1          ; p1 = false for lane 0

    wave_all p2, p1

    mov_imm r2, 0
    mov_imm r3, 1
    mov_imm r4, 0
    @p2 device_store_u32 r2, r3
    @!p2 device_store_u32 r2, r4

    ; Test 2: All lanes true
    mov_imm r1, 100
    icmp_lt p1, r0, r1          ; p1 = true for all (lane_id < 100)

    wave_all p2, p1

    mov_imm r2, 4
    @p2 device_store_u32 r2, r3
    @!p2 device_store_u32 r2, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let some_false = read_u32(&result.device_memory, 0);  // Should be 0
            let all_true = read_u32(&result.device_memory, 4);    // Should be 1

            let passed = some_false == 0 && all_true == 1;

            TestResult {
                name: "test_wave_all".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_all checks all predicates".to_string(),
                passed,
                details: format!(
                    "some_false={}, all_true={} (expected 0, 1)",
                    some_false, all_true
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_all".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_all checks all predicates".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Wave reduce add sums all lane values.
fn test_wave_reduce_add() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_reduce_add sums values across all lanes."

.kernel test_wave_reduce_add
.registers 8
    mov r0, sr_lane_id

    ; Each lane has value = lane_id + 1 (so 1, 2, 3, 4)
    mov_imm r1, 1
    iadd r2, r0, r1

    wave_reduce_add r3, r2      ; r3 = 1 + 2 + 3 + 4 = 10

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
                name: "test_wave_reduce_add".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_reduce_add sums correctly".to_string(),
                passed,
                details: format!("sum(1,2,3,4) = {} (expected 10)", sum),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_reduce_add".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_reduce_add sums correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Wave prefix sum computes exclusive prefix sum.
fn test_wave_prefix_sum() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.12 Wave Operations
; Claim: "wave_prefix_sum computes exclusive prefix sum."

.kernel test_wave_prefix_sum
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Each lane has value 1
    mov_imm r3, 1

    wave_prefix_sum r4, r3      ; Lane 0: 0, Lane 1: 1, Lane 2: 2, Lane 3: 3

    device_store_u32 r2, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);
            let val1 = read_u32(&result.device_memory, 4);
            let val2 = read_u32(&result.device_memory, 8);
            let val3 = read_u32(&result.device_memory, 12);

            let passed = val0 == 0 && val1 == 1 && val2 == 2 && val3 == 3;

            TestResult {
                name: "test_wave_prefix_sum".to_string(),
                spec_section: "6.12".to_string(),
                spec_claim: "wave_prefix_sum computes correctly".to_string(),
                passed,
                details: format!(
                    "prefix_sum=[{}, {}, {}, {}] (expected [0, 1, 2, 3])",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_wave_prefix_sum".to_string(),
            spec_section: "6.12".to_string(),
            spec_claim: "wave_prefix_sum computes correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
