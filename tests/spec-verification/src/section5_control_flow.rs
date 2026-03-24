// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Section 5: Control Flow tests
//!
//! Verifies divergence, reconvergence, loops, and inactive thread semantics.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_if_uniform_taken(),
        test_if_uniform_not_taken(),
        test_if_else_uniform(),
        test_if_divergent(),
        test_if_else_divergent(),
        test_nested_if_2_levels(),
        test_reconvergence_after_endif(),
        test_inactive_no_side_effects(),
        test_loop_simple(),
        test_loop_break(),
        test_loop_divergent_break(),
        test_halt_partial_wave(),
    ]
}

/// Spec Section 5.1: Uniform branch - all threads agree, take the branch.
fn test_if_uniform_taken() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "When all threads agree on branch, only one path executes."
; Method: All threads have same condition (true). Verify all execute if-body.

.kernel test_if_uniform_taken
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Set p1 = (1 == 1) = true for all threads
    mov_imm r3, 1
    mov_imm r4, 1
    icmp_eq p1, r3, r4

    ; All threads should execute this
    if p1
        mov_imm r5, 42
        device_store_u32 r2, r5
    endif

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
                name: "test_if_uniform_taken".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Uniform branch (taken) executes correctly".to_string(),
                passed,
                details: if passed {
                    "All 4 threads executed if-body".to_string()
                } else {
                    format_memory_u32(&result.device_memory, 0, 4)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_if_uniform_taken".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Uniform branch (taken) executes correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Uniform branch - all threads agree, skip the branch.
fn test_if_uniform_not_taken() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "When all threads agree on branch, only one path executes."
; Method: All threads have same condition (false). Verify none execute if-body.

.kernel test_if_uniform_not_taken
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Set p1 = (1 == 2) = false for all threads
    mov_imm r3, 1
    mov_imm r4, 2
    icmp_eq p1, r3, r4

    ; No threads should execute this
    if p1
        mov_imm r5, 42
        device_store_u32 r2, r5
    endif

    ; Write completion marker
    mov_imm r5, 0xDEADBEEF
    mov_imm r6, 100
    device_store_u32 r6, r5

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let mut if_executed = false;
            for lane in 0..4 {
                let value = read_u32(&result.device_memory, lane * 4);
                if value == 42 {
                    if_executed = true;
                    break;
                }
            }
            let completion = read_u32(&result.device_memory, 100);
            let passed = !if_executed && completion == COMPLETION_FLAG;

            TestResult {
                name: "test_if_uniform_not_taken".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Uniform branch (not taken) skips correctly".to_string(),
                passed,
                details: if passed {
                    "No threads executed if-body".to_string()
                } else {
                    "Some threads incorrectly executed if-body".to_string()
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_if_uniform_not_taken".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Uniform branch (not taken) skips correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: If-else with uniform condition.
fn test_if_else_uniform() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "if/else with uniform condition executes exactly one branch."

.kernel test_if_else_uniform
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; p1 = false (uniform)
    mov_imm r3, 1
    mov_imm r4, 2
    icmp_eq p1, r3, r4

    if p1
        mov_imm r5, 10
        device_store_u32 r2, r5
    else
        mov_imm r5, 20
        device_store_u32 r2, r5
    endif

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
                if value != 20 {  // All should have 20 (else branch)
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_if_else_uniform".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "If-else uniform executes else branch".to_string(),
                passed,
                details: if passed {
                    "All threads executed else branch".to_string()
                } else {
                    format_memory_u32(&result.device_memory, 0, 4)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_if_else_uniform".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "If-else uniform executes else branch".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Divergent if - threads disagree on condition.
fn test_if_divergent() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "When threads disagree, both paths execute with appropriate masks."
; Method: Even lanes take if, odd lanes skip. Verify only even lanes write.

.kernel test_if_divergent
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; p1 = (lane_id & 1) == 0 (true for even lanes)
    mov_imm r3, 1
    and r4, r0, r3              ; r4 = lane_id & 1
    mov_imm r5, 0
    icmp_eq p1, r4, r5          ; p1 = true if even

    if p1
        mov_imm r6, 100
        iadd r6, r6, r0         ; value = 100 + lane_id
        device_store_u32 r2, r6
    endif

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

            let passed = val0 == 100 && val1 == 0 && val2 == 102 && val3 == 0;

            TestResult {
                name: "test_if_divergent".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Divergent if executes only for active threads".to_string(),
                passed,
                details: format!(
                    "lanes=[{}, {}, {}, {}] (expected [100, 0, 102, 0])",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_if_divergent".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Divergent if executes only for active threads".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Divergent if-else - both paths execute.
fn test_if_else_divergent() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "When threads disagree, both paths execute."
; Method: Even lanes write 1, odd lanes write 2.

.kernel test_if_else_divergent
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; p1 = (lane_id & 1) == 0 (true for even lanes)
    mov_imm r3, 1
    and r4, r0, r3
    mov_imm r5, 0
    icmp_eq p1, r4, r5

    if p1
        mov_imm r6, 1
        device_store_u32 r2, r6
    else
        mov_imm r6, 2
        device_store_u32 r2, r6
    endif

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);   // even -> 1
            let val1 = read_u32(&result.device_memory, 4);   // odd  -> 2
            let val2 = read_u32(&result.device_memory, 8);   // even -> 1
            let val3 = read_u32(&result.device_memory, 12);  // odd  -> 2

            let passed = val0 == 1 && val1 == 2 && val2 == 1 && val3 == 2;

            TestResult {
                name: "test_if_else_divergent".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Divergent if-else executes both paths".to_string(),
                passed,
                details: format!(
                    "lanes=[{}, {}, {}, {}] (expected [1, 2, 1, 2])",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_if_else_divergent".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Divergent if-else executes both paths".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Nested if statements.
fn test_nested_if_2_levels() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "Control flow can be nested."
; Method: Outer if on bit 0, inner if on bit 1. 4 possible outcomes.

.kernel test_nested_if_2_levels
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Outer: p1 = (lane_id & 1) == 0
    mov_imm r3, 1
    and r4, r0, r3
    mov_imm r5, 0
    icmp_eq p1, r4, r5

    ; Initialize result
    mov_imm r6, 0

    if p1
        ; lane 0 and 2 enter here
        ; Inner: p2 = (lane_id & 2) == 0
        mov_imm r3, 2
        and r4, r0, r3
        icmp_eq p2, r4, r5

        if p2
            ; Only lane 0: bit0=0, bit1=0
            mov_imm r6, 10
        else
            ; Only lane 2: bit0=0, bit1=1
            mov_imm r6, 20
        endif
    else
        ; lane 1 and 3 enter here
        mov_imm r3, 2
        and r4, r0, r3
        icmp_eq p2, r4, r5

        if p2
            ; Only lane 1: bit0=1, bit1=0
            mov_imm r6, 30
        else
            ; Only lane 3: bit0=1, bit1=1
            mov_imm r6, 40
        endif
    endif

    device_store_u32 r2, r6

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);   // lane 0: 10
            let val1 = read_u32(&result.device_memory, 4);   // lane 1: 30
            let val2 = read_u32(&result.device_memory, 8);   // lane 2: 20
            let val3 = read_u32(&result.device_memory, 12);  // lane 3: 40

            let passed = val0 == 10 && val1 == 30 && val2 == 20 && val3 == 40;

            TestResult {
                name: "test_nested_if_2_levels".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Nested if works correctly".to_string(),
                passed,
                details: format!(
                    "lanes=[{}, {}, {}, {}] (expected [10, 30, 20, 40])",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_nested_if_2_levels".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Nested if works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: After endif, all threads that were active before are active again.
fn test_reconvergence_after_endif() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "After endif, all threads that were active before if are active again."
; Method: Diverge, then verify all threads write after endif.

.kernel test_reconvergence_after_endif
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Diverge on even/odd
    mov_imm r3, 1
    and r4, r0, r3
    mov_imm r5, 0
    icmp_eq p1, r4, r5

    if p1
        nop                     ; Even lanes do nothing interesting
    else
        nop                     ; Odd lanes do nothing interesting
    endif

    ; After endif, ALL threads should be active
    mov_imm r6, 0xABCD
    device_store_u32 r2, r6

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
                if value != 0xABCD {
                    passed = false;
                    break;
                }
            }

            TestResult {
                name: "test_reconvergence_after_endif".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Threads reconverge after endif".to_string(),
                passed,
                details: if passed {
                    "All 4 threads wrote after endif".to_string()
                } else {
                    format_memory_u32(&result.device_memory, 0, 4)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_reconvergence_after_endif".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Threads reconverge after endif".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Inactive threads produce no side effects.
fn test_inactive_no_side_effects() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "Inactive threads producing no architectural side effects."
; Method: Diverge. In if-body, ALL threads have a store in instruction stream,
;         but inactive threads' stores should NOT execute.

.kernel test_inactive_no_side_effects
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Only lane 0 is active in if-body
    mov_imm r3, 0
    icmp_eq p1, r0, r3

    if p1
        ; This instruction is in the stream for all threads,
        ; but only lane 0 should actually store
        mov_imm r6, 0x12345678
        device_store_u32 r2, r6
    endif

    ; Write completion flag at offset 100
    mov_imm r5, 100
    mov_imm r6, 0xDEADBEEF
    device_store_u32 r5, r6

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);   // Lane 0: should write
            let val1 = read_u32(&result.device_memory, 4);   // Lane 1: should NOT write
            let val2 = read_u32(&result.device_memory, 8);   // Lane 2: should NOT write
            let val3 = read_u32(&result.device_memory, 12);  // Lane 3: should NOT write
            let completion = read_u32(&result.device_memory, 100);

            let passed = val0 == 0x12345678 && val1 == 0 && val2 == 0 && val3 == 0
                && completion == COMPLETION_FLAG;

            TestResult {
                name: "test_inactive_no_side_effects".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Inactive threads produce no side effects".to_string(),
                passed,
                details: format!(
                    "lanes=[0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}]",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_inactive_no_side_effects".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Inactive threads produce no side effects".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Simple loop.
fn test_loop_simple() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "loop/endloop creates a loop."
; Method: Loop 5 times, accumulate sum.

.kernel test_loop_simple
.registers 8
    mov_imm r0, 0               ; counter
    mov_imm r1, 0               ; sum
    mov_imm r2, 5               ; limit

    loop
        iadd r1, r1, r0         ; sum += counter
        mov_imm r3, 1
        iadd r0, r0, r3         ; counter++
        icmp_ge p1, r0, r2      ; p1 = counter >= 5
        break p1                ; break if counter >= 5
    endloop

    ; Write sum to device memory (should be 0+1+2+3+4 = 10)
    mov_imm r4, 0
    device_store_u32 r4, r1

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let sum = read_u32(&result.device_memory, 0);
            let passed = sum == 10;

            TestResult {
                name: "test_loop_simple".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Simple loop works correctly".to_string(),
                passed,
                details: format!("Sum = {} (expected 10)", sum),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_loop_simple".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Simple loop works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Loop with break.
fn test_loop_break() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "break exits the innermost loop."

.kernel test_loop_break
.registers 8
    mov_imm r0, 0               ; counter

    loop
        mov_imm r1, 1
        iadd r0, r0, r1         ; counter++

        ; Break when counter reaches 3
        mov_imm r2, 3
        icmp_eq p1, r0, r2
        break p1
    endloop

    ; Write counter (should be 3)
    mov_imm r3, 0
    device_store_u32 r3, r0

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let counter = read_u32(&result.device_memory, 0);
            let passed = counter == 3;

            TestResult {
                name: "test_loop_break".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "break exits loop correctly".to_string(),
                passed,
                details: format!("Counter = {} (expected 3)", counter),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_loop_break".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "break exits loop correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: Divergent break - threads exit at different iterations.
fn test_loop_divergent_break() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "Loop executes until all active threads have exited via break."
; Method: Each thread breaks when counter equals its lane_id.

.kernel test_loop_divergent_break
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    mov_imm r3, 0               ; counter

    loop
        ; Break when counter == lane_id
        icmp_eq p1, r3, r0
        break p1

        mov_imm r4, 1
        iadd r3, r3, r4
    endloop

    ; Each thread writes its exit counter (should equal lane_id)
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
            let val0 = read_u32(&result.device_memory, 0);   // Lane 0 breaks at 0
            let val1 = read_u32(&result.device_memory, 4);   // Lane 1 breaks at 1
            let val2 = read_u32(&result.device_memory, 8);   // Lane 2 breaks at 2
            let val3 = read_u32(&result.device_memory, 12);  // Lane 3 breaks at 3

            let passed = val0 == 0 && val1 == 1 && val2 == 2 && val3 == 3;

            TestResult {
                name: "test_loop_divergent_break".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Divergent break works correctly".to_string(),
                passed,
                details: format!(
                    "lanes=[{}, {}, {}, {}] (expected [0, 1, 2, 3])",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_loop_divergent_break".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Divergent break works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 5.1: halt terminates threads, others continue.
fn test_halt_partial_wave() -> TestResult {
    const SOURCE: &str = r#"
; Section: 5.1 Control Flow
; Claim: "halt terminates the thread. Wave continues with remaining active threads."
; Method: Lane 0 halts early. Other lanes continue and write.

.kernel test_halt_partial_wave
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1              ; offset = lane_id * 4

    ; Lane 0 halts immediately
    mov_imm r3, 0
    icmp_eq p1, r0, r3
    @p1 mov_imm r4, 0xDEAD
    @p1 device_store_u32 r2, r4
    @p1 halt

    ; Other lanes continue
    mov_imm r4, 0xBEEF
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
            let val0 = read_u32(&result.device_memory, 0);   // Lane 0: 0xDEAD (halted early)
            let val1 = read_u32(&result.device_memory, 4);   // Lane 1: 0xBEEF
            let val2 = read_u32(&result.device_memory, 8);   // Lane 2: 0xBEEF
            let val3 = read_u32(&result.device_memory, 12);  // Lane 3: 0xBEEF

            let passed = val0 == 0xDEAD && val1 == 0xBEEF && val2 == 0xBEEF && val3 == 0xBEEF;

            TestResult {
                name: "test_halt_partial_wave".to_string(),
                spec_section: "5.1".to_string(),
                spec_claim: "Partial halt allows other threads to continue".to_string(),
                passed,
                details: format!(
                    "lanes=[0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}]",
                    val0, val1, val2, val3
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_halt_partial_wave".to_string(),
            spec_section: "5.1".to_string(),
            spec_claim: "Partial halt allows other threads to continue".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
